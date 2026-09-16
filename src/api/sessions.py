# src/api/sessions.py
"""
In-memory conversation session store.

Each session keeps the user/assistant turns of one conversation, the ids
of the chunks that produced the previous answer (so "more information"
follow-ups can exclude them), and a last-access timestamp.  Sessions idle
for longer than ``SESSION_TTL`` are pruned on the next request.

The store is intentionally process-local: a single API worker owns the
conversations.  Swapping it for Redis later only means re-implementing
these functions.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

SESSION_TTL = 3600  # 1 hour — sessions older than this are pruned

# session_id → {"history": [{"role": ..., "content": ...}, ...],
#               "chunk_ids": [ids of chunks used in the last turn],
#               "last_access": float}
sessions: dict[str, dict] = {}


def get_history(session_id: str | None) -> list[dict[str, str]]:
    """Retrieve conversation history for a session (empty list if new)."""
    if not session_id or session_id not in sessions:
        return []
    sessions[session_id]["last_access"] = time.time()
    return sessions[session_id]["history"]


def get_used_chunk_ids(session_id: str | None) -> list[str]:
    """Ids of the chunks that produced the previous answer in this session.

    Passed back to the QA system so that a follow-up asking for additional
    information can exclude them and retrieve new chunks.
    """
    if not session_id or session_id not in sessions:
        return []
    return list(sessions[session_id].get("chunk_ids") or [])


def clean_answer(answer: str) -> str:
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


def save_turn(
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
    hist.append({"role": "assistant", "content": clean_answer(answer)})
    sessions[session_id]["chunk_ids"] = list(chunk_ids or [])
    sessions[session_id]["last_access"] = time.time()


def prune_sessions() -> None:
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


def active_session_count() -> int:
    """Number of sessions currently held in memory."""
    return len(sessions)
