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
from src.chains.qa_chain import UniversityQASystem
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
#                 "last_access": float}
# ------------------------------------------------------------------ #

SESSION_TTL = 3600  # 1 hour — sessions older than this are pruned

sessions: dict[str, dict] = {}


def _get_history(session_id: str | None) -> list[dict[str, str]]:
    """Retrieve conversation history for a session (empty list if new)."""
    if not session_id or session_id not in sessions:
        return []
    sessions[session_id]["last_access"] = time.time()
    return sessions[session_id]["history"]


def _clean_answer(answer: str) -> str:
    """Strip the trailing source citation block so it does not pollute
    the conversation history that is fed back to the LLM on future turns."""
    idx = answer.rfind("\n\n---\n")
    if idx != -1:
        return answer[:idx].strip()
    return answer


def _save_turn(
    session_id: str | None,
    question: str,
    answer: str,
) -> None:
    """Append a user/assistant turn to the session history."""
    if not session_id:
        return
    if session_id not in sessions:
        sessions[session_id] = {"history": [], "last_access": time.time()}
    hist = sessions[session_id]["history"]
    hist.append({"role": "user", "content": question})
    hist.append({"role": "assistant", "content": _clean_answer(answer)})
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
      1. Retrieve conversation history for session.
      2. Stream tokens from QA system.
      3. Accumulate the full answer.
      4. Save the completed turn to session history.
    """
    history = _get_history(session_id)
    _prune_sessions()  # occasional cleanup

    accumulated: list[str] = []

    try:
        async for token in qa_system.query_stream(question, history):
            accumulated.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"

        full_answer = "".join(accumulated)
        _save_turn(session_id, question, full_answer)

        yield f"data: {json.dumps({'done': True})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


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