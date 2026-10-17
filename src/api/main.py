# src/api/main.py
"""
FastAPI application for the University Chatbot.

v2 — Multi-turn conversation support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Each request carries an optional ``session_id``.
- The server keeps a conversation history per session — in memory, or in
  Redis when REDIS_URL is set (see sessions.py).
- Every turn: if the question refers back to an earlier answer ("njih",
  "o njima", …) rewrite it into a standalone question using the history →
  embed → search → send ONLY system prompt + retrieved chunks + question
  to Ollama /api/chat (no history) → stream tokens + sources → save turn
  to history.
- /chat/stream is limited to RATE_LIMIT requests per minute per client IP
  (see middleware.py).
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config.settings import settings
from src.api.middleware import RateLimiter
from src.api.request_log import RequestLogger, build_record, new_request_id
from src.api.sessions import MemorySessionStore, RedisSessionStore, connect_redis
from src.core.vector_store import VectorStoreManager
from src.core.llm_manager import LLMManager
from src.chains.qa_chain import UniversityQASystem
from src.processing.translator import ScriptTranslator
from src.processing.document_loader import DocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Build the QA system once, in the process that serves the requests.

    This used to run at import time, so every process that imported this
    module loaded the embedding model — the launcher, the reloader and the
    worker, i.e. several copies of it under `reload=True`.
    """
    global qa_system
    qa_system = initialize_system()
    yield


app = FastAPI(title="University Chatbot API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
#  Sessions + rate limiting                                           #
# ------------------------------------------------------------------ #
#  Kept in Redis when REDIS_URL is set (shared by all workers, kept
#  across restarts), otherwise in this process's memory.
# ------------------------------------------------------------------ #

redis_client  = connect_redis(settings.REDIS_URL)
session_store = RedisSessionStore(redis_client) if redis_client else MemorySessionStore()
rate_limiter  = RateLimiter(settings.RATE_LIMIT, redis=redis_client)

# Structured per-turn trace for the visualization project (see request_log.py).
# The chatbot only ever writes this file; nothing reads it back.
request_log = RequestLogger(
    settings.REQUEST_LOG_PATH,
    enabled=settings.REQUEST_LOG_ENABLED,
    max_bytes=settings.REQUEST_LOG_MAX_BYTES,
    backups=settings.REQUEST_LOG_BACKUPS,
)


async def _check_rate_limit(request: Request) -> None:
    """Dependency for /chat routes: HTTP 429 once a client IP exceeds RATE_LIMIT."""
    client = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client):
        raise HTTPException(
            status_code=429,
            detail="Previše zahteva. Pokušajte ponovo za minut.",
            headers={"Retry-After": str(rate_limiter.retry_after())},
        )


def _clean_answer(answer: str) -> str:
    """Strip the trailing source citation block so it does not pollute
    the conversation history that the query-rewrite step reads on future
    turns."""
    idx = answer.rfind("\n\n---\n")
    if idx != -1:
        return answer[:idx].strip()
    return answer


# ------------------------------------------------------------------ #
#  Models                                                             #
# ------------------------------------------------------------------ #

class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None   # <-- NEW: pass to enable multi-turn
    script: str = "latinic"         # cyrillic or latinic
    language: str = "sr"            # sr, mk, bg


class ChatResponse(BaseModel):
    question: str
    answer: str
    # sources: list
    # confidence: float


# ------------------------------------------------------------------ #
#  System initialisation                                              #
# ------------------------------------------------------------------ #

def initialize_system() -> UniversityQASystem:
    """Initialize all components."""
    logger.info("Initializing system components...")

    translator = ScriptTranslator()
    llm_manager = LLMManager(settings)
    vector_store = VectorStoreManager(settings)

    # Document processor (kept for ad-hoc indexing if needed)
    _doc_processor = DocumentProcessor(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    qa_system = UniversityQASystem(llm_manager, vector_store, translator)
    logger.info("System initialized successfully!")
    return qa_system


# Filled in by the lifespan handler on startup; stays None when the module is
# only imported (tooling, type checks), so importing it no longer loads models.
qa_system: UniversityQASystem | None = None


# ------------------------------------------------------------------ #
#  Routes                                                             #
# ------------------------------------------------------------------ #

@app.post("/chat/stream", dependencies=[Depends(_check_rate_limit)])
async def chat_stream(payload: ChatRequest, request: Request):
    """Stream response token by token (SSE)."""
    try:
        return StreamingResponse(
            _stream_response(payload.question, payload.session_id, request),
            media_type="text/event-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_response(
    question: str,
    session_id: str | None = None,
    request: Request | None = None,
) -> AsyncGenerator[str, None]:
    """
    Core streaming logic:
      1. Retrieve conversation history for session.
      2. Stream tokens from QA system.
      3. Accumulate the full answer.
      4. Save the completed turn to session history.
      5. Append one trace record to the request log.

    The trace dict is filled in by the QA system as it works (rewrite,
    retrieval, prompt) and written once, at the end, when the answer and the
    timings are known.  A failed turn is logged too, with ``failed: true``.
    """
    accumulated: list[str] = []
    trace: dict = {}
    request_id = new_request_id()
    started = time.perf_counter()
    error: str | None = None
    history: list[dict[str, str]] = []

    def log_turn() -> None:
        """
        Append this turn to the request log.  Never raises: the log is a
        debugging aid, so a failure to write it must not affect the answer
        the client already received.
        """
        if request is None:  # pragma: no cover - only in direct calls/tests
            return
        try:
            request_log.write(build_record(
                request_id=request_id,
                session_id=session_id,
                endpoint=str(request.url.path),
                client=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                question=question,
                trace=trace,
                answer="".join(accumulated),
                history_turns=len(history),
                duration_ms=(time.perf_counter() - started) * 1000,
                max_text=settings.REQUEST_LOG_MAX_TEXT,
                max_excluded=settings.REQUEST_LOG_MAX_EXCLUDED,
                error=error,
            ))
        except Exception:  # pragma: no cover - defensive
            logger.exception("Could not build the request log record")

    if qa_system is None:  # pragma: no cover - startup always initialises it
        error = "QA system is not initialized"
        yield f"data: {json.dumps({'error': error})}\n\n"
        log_turn()
        return

    try:
        history = session_store.get_history(session_id)
        async for token in qa_system.query_stream(question, history, trace):
            accumulated.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"

        full_answer = "".join(accumulated)
        session_store.save_turn(session_id, question, _clean_answer(full_answer))

        yield f"data: {json.dumps({'done': True})}\n\n"

    except Exception as e:
        error = str(e)
        logger.exception("Turn %s failed", request_id)
        yield f"data: {json.dumps({'error': error})}\n\n"

    finally:
        # Also on a client that hung up mid-answer: the record then holds the
        # pipeline steps that did run, which is exactly how such a turn is
        # diagnosed.
        log_turn()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "active_sessions": session_store.active_count()}


if __name__ == "__main__":
    import uvicorn

    # Fully-qualified import string: `reload=True` makes uvicorn import the
    # app in a child process, where a bare "main:app" is not importable —
    # this module lives at src/api/main.py, so the project root is the only
    # place "main" resolves from.  `python -m src.api.main` and
    # `python src/api/main.py` both work with the qualified name.
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )