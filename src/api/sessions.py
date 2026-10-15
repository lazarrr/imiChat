"""
sessions.py
~~~~~~~~~~~
Per-session conversation history for the chat API.

The history is only read by the query-rewrite step (unclear follow-ups
such as "a o njima?"); it is never sent to the answering LLM.

Two interchangeable stores:
  MemorySessionStore — the API process's memory (default).  Lost on
                       restart and not shared between workers.
  RedisSessionStore  — used when REDIS_URL is set.  Shared by all
                       workers and kept across restarts.
Both forget a session after SESSION_TTL seconds without activity.
"""

from __future__ import annotations

import json
import logging
import time

logger = logging.getLogger(__name__)

SESSION_TTL = 3600  # 1 hour — sessions idle longer than this are forgotten


def connect_redis(url: str | None):
    """
    Redis client for ``url``, or None (in-memory fallback) when no URL is
    configured, the ``redis`` package is missing or the server does not
    answer.
    """
    if not url:
        return None
    try:
        import redis  # optional dependency — only needed when REDIS_URL is set

        client = redis.Redis.from_url(
            url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2
        )
        client.ping()
    except Exception as e:
        logger.warning(
            "REDIS_URL is set but Redis is not available (%s) — "
            "keeping sessions and rate limits in memory", e,
        )
        return None
    logger.info("Using Redis for chat sessions and rate limits")
    return client


class MemorySessionStore:
    """Sessions in this process's memory (lost on restart, not shared by workers)."""

    def __init__(self, ttl: int = SESSION_TTL):
        self.ttl = ttl
        # session_id → {"history": [{"role": ..., "content": ...}, ...],
        #               "last_access": float}
        self._sessions: dict[str, dict] = {}

    def get_history(self, session_id: str | None) -> list[dict[str, str]]:
        """Conversation history for a session (empty list if new)."""
        self._prune()
        session = self._sessions.get(session_id) if session_id else None
        if session is None:
            return []
        session["last_access"] = time.time()
        return session["history"]

    def save_turn(self, session_id: str | None, question: str, answer: str) -> None:
        """Append a user/assistant turn to the session history."""
        if not session_id:
            return
        session = self._sessions.setdefault(session_id, {"history": []})
        session["history"] += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        session["last_access"] = time.time()

    def active_count(self) -> int:
        self._prune()
        return len(self._sessions)

    def _prune(self) -> None:
        """Remove sessions that have been idle longer than the TTL."""
        now = time.time()
        stale = [
            sid for sid, s in self._sessions.items()
            if now - s.get("last_access", 0) > self.ttl
        ]
        for sid in stale:
            del self._sessions[sid]
        if stale:
            logger.info("Pruned %d stale session(s)", len(stale))


class RedisSessionStore:
    """
    Sessions in Redis — one list per session, one JSON message per item.
    Redis itself expires sessions that stay idle longer than the TTL.
    """

    KEY_PREFIX = "imichat:session:"

    def __init__(self, redis, ttl: int = SESSION_TTL):
        self.redis = redis
        self.ttl = ttl

    def get_history(self, session_id: str | None) -> list[dict[str, str]]:
        """Conversation history for a session (empty list if new)."""
        if not session_id:
            return []
        key = self.KEY_PREFIX + session_id
        items = self.redis.lrange(key, 0, -1)
        if items:
            self.redis.expire(key, self.ttl)  # activity keeps the session alive
        return [json.loads(item) for item in items]

    def save_turn(self, session_id: str | None, question: str, answer: str) -> None:
        """Append a user/assistant turn to the session history."""
        if not session_id:
            return
        key = self.KEY_PREFIX + session_id
        pipe = self.redis.pipeline()
        pipe.rpush(
            key,
            json.dumps({"role": "user", "content": question}, ensure_ascii=False),
            json.dumps({"role": "assistant", "content": answer}, ensure_ascii=False),
        )
        pipe.expire(key, self.ttl)
        pipe.execute()

    def active_count(self) -> int:
        return sum(1 for _ in self.redis.scan_iter(match=self.KEY_PREFIX + "*", count=1000))
