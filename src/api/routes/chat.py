# src/api/routes/chat.py
"""
Streaming chat endpoint (Server-Sent Events) with multi-turn sessions.

Every turn: load the session history and the chunks used previously →
stream tokens from the QA system (citations validated inline) → emit a
structured ``sources`` event → save the completed turn → emit ``done``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.dependencies import qa_system
from src.api.schemas import ChatRequest
from src.api.sessions import (
    get_history,
    get_used_chunk_ids,
    prune_sessions,
    save_turn,
)
from src.chains.qa_chain import (
    RetrievedIdsChunk,
    SourcesChunk,
    TokenChunk,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _sse(payload: dict) -> str:
    """Encode a payload as a Server-Sent Events frame (UTF-8, unescaped)."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_response(
    question: str,
    session_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream one answer as SSE frames and persist the turn afterwards."""
    history = get_history(session_id)
    # Chunks used in the previous turn — a "more information" follow-up
    # excludes them so the answer model gets fresh material instead.
    exclude_chunk_ids = get_used_chunk_ids(session_id)
    prune_sessions()  # occasional cleanup

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
        save_turn(session_id, question, full_answer, used_chunk_ids)

        yield _sse({"done": True})

    except Exception as e:
        logger.exception("Streaming failed")
        yield _sse({"error": str(e)})


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream response token by token (SSE)."""
    try:
        return StreamingResponse(
            _stream_response(request.question, request.session_id),
            media_type="text/event-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
