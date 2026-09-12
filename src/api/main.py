# src/api/main.py
"""
FastAPI application for the University Chatbot.

v2 — Multi-turn conversation support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Each request carries an optional ``session_id``.
- The server maintains an in-memory conversation history per session.
- Every turn: rewrite the question for retrieval → embed → search →
  build full message list (system + history + context) → call
  Ollama /api/chat → stream tokens + sources → save turn to history.
"""

from __future__ import annotations

import json
import time
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config.settings import settings
from src.core.vector_store import VectorStoreManager
from src.core.llm_manager import LLMManager
from src.chains.qa_chain import (
    RetrievedIdsChunk,
    SourcesChunk,
    TokenChunk,
    UniversityQASystem,
)
from src.processing.translator import ScriptTranslator
from src.processing.document_loader import DocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="University Chatbot API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
#  In-memory session store                                            #
# ------------------------------------------------------------------ #
#  session_id → {"history": [{"role": ..., "content": ...}, ...],
#                "chunk_ids": [ids of chunks used in the last turn],
#                "last_access": float}
# ------------------------------------------------------------------ #

SESSION_TTL = 3600  # 1 hour — sessions older than this are pruned

sessions: dict[str, dict] = {}


def _get_history(session_id: str | None) -> list[dict[str, str]]:
    """Retrieve conversation history for a session (empty list if new)."""
    if not session_id or session_id not in sessions:
        return []
    sessions[session_id]["last_access"] = time.time()
    return sessions[session_id]["history"]


def _get_used_chunk_ids(session_id: str | None) -> list[str]:
    """Ids of the chunks that produced the previous answer in this session.

    Passed back to the QA system so that a follow-up asking for additional
    information can exclude them and retrieve new chunks.
    """
    if not session_id or session_id not in sessions:
        return []
    return list(sessions[session_id].get("chunk_ids") or [])


def _clean_answer(answer: str) -> str:
    """Strip a trailing ``Izvori:`` citation block so it does not pollute
    the conversation history that is fed back to the LLM on future turns.

    The streaming path no longer appends the block (sources travel in a
    dedicated SSE event), but old sessions and the non-streaming path may
    still contain one.
    """
    marker = "\n\n---\nIzvori:"
    idx = answer.rfind(marker)
    if idx != -1:
        return answer[:idx].strip()
    return answer


def _sse(payload: dict) -> str:
    """Encode a payload as a Server-Sent Events frame (UTF-8, unescaped)."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _save_turn(
    session_id: str | None,
    question: str,
    answer: str,
    chunk_ids: list[str] | None = None,
) -> None:
    """Append a user/assistant turn to the session history."""
    if not session_id:
        return
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "chunk_ids": [],
            "last_access": time.time(),
        }
    hist = sessions[session_id]["history"]
    hist.append({"role": "user", "content": question})
    hist.append({"role": "assistant", "content": _clean_answer(answer)})
    sessions[session_id]["chunk_ids"] = list(chunk_ids or [])
    sessions[session_id]["last_access"] = time.time()


def _prune_sessions() -> None:
    """Remove sessions that have exceeded SESSION_TTL."""
    now = time.time()
    stale = [
        sid for sid, s in sessions.items()
        if now - s.get("last_access", 0) > SESSION_TTL
    ]
    for sid in stale:
        del sessions[sid]
    if stale:
        logger.info("Pruned %d stale session(s)", len(stale))


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


qa_system = initialize_system()


# ------------------------------------------------------------------ #
#  Routes                                                             #
# ------------------------------------------------------------------ #

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream response token by token (SSE)."""
    try:
        return StreamingResponse(
            _stream_response(request.question, request.session_id),
            media_type="text/event-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_response(
    question: str,
    session_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Core streaming logic:
      1. Retrieve conversation history + previously used chunk ids.
      2. Stream tokens from the QA system (citations validated inline).
      3. Emit a structured ``sources`` event.
      4. Save the completed turn (with its chunk ids) to session history.
      5. Emit ``done``.
    """
    history = _get_history(session_id)
    exclude_chunk_ids = _get_used_chunk_ids(session_id)
    _prune_sessions()  # occasional cleanup

    accumulated: list[str] = []
    used_chunk_ids: list[str] = []

    try:
        async for chunk in qa_system.query_stream(
            question, history, exclude_chunk_ids=exclude_chunk_ids
        ):
            if isinstance(chunk, TokenChunk):
                accumulated.append(chunk.text)
                yield _sse({"token": chunk.text})
            elif isinstance(chunk, SourcesChunk):
                yield _sse({"sources": chunk.sources})
            elif isinstance(chunk, RetrievedIdsChunk):
                used_chunk_ids = list(chunk.chunk_ids)

        full_answer = "".join(accumulated)
        _save_turn(session_id, question, full_answer, used_chunk_ids)

        yield _sse({"done": True})

    except Exception as e:
        logger.exception("Streaming failed")
        yield _sse({"error": str(e)})


@app.get("/health")
async def health_check():
    return {"status": "healthy", "active_sessions": len(sessions)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )