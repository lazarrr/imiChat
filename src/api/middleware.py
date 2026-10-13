"""
middleware.py
~~~~~~~~~~~~~
Rate limiting for the chat API: at most RATE_LIMIT /chat requests per
client IP per minute (0 = no limit).  main.py applies it as a FastAPI
dependency on /chat/stream, so /health and CORS pre-flight requests are
never limited, and a refused request gets HTTP 429.

Counters use fixed one-minute windows.  They live in Redis when REDIS_URL
is set (one shared limit across all API workers), otherwise in this
process's memory.

Behind a reverse proxy, run uvicorn with --proxy-headers (and
--forwarded-allow-ips=<proxy IP>) so the limit applies to each student's
IP rather than to the proxy's.
"""

from __future__ import annotations

import time


class RateLimiter:
    KEY_PREFIX = "imichat:ratelimit:"

    def __init__(self, limit: int, window: int = 60, redis=None):
        self.limit  = limit   # requests per window per client; 0 = no limit
        self.window = window  # seconds
        self.redis  = redis
        self._current_window = -1
        self._counts: dict[str, int] = {}

    def allow(self, client: str) -> bool:
        """Count one request from ``client``; False once it is over the limit."""
        if self.limit <= 0:
            return True
        window = int(time.time()) // self.window

        if self.redis is not None:
            key = f"{self.KEY_PREFIX}{client}:{window}"
            count = self.redis.incr(key)
            if count == 1:
                self.redis.expire(key, self.window)
            return count <= self.limit

        if window != self._current_window:  # new window: forget the old counts
            self._current_window, self._counts = window, {}
        self._counts[client] = self._counts.get(client, 0) + 1
        return self._counts[client] <= self.limit

    def retry_after(self) -> int:
        """Seconds until the current window ends and requests are allowed again."""
        return self.window - int(time.time()) % self.window
