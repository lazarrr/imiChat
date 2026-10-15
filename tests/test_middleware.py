"""
Tests for the chat rate limiter (src/api/middleware.py).

    python -m unittest tests.test_middleware -v
"""

import unittest
from unittest.mock import patch

from src.api.middleware import RateLimiter

from .fake_redis import FakeRedis


def at(seconds: float):
    """Freeze the limiter's clock at ``seconds``."""
    return patch("src.api.middleware.time.time", return_value=seconds)


class RateLimiterContract:
    """Behaviour shared by the in-memory and Redis counters."""

    def test_allows_up_to_the_limit_then_refuses(self):
        limiter = self.make_limiter(3)
        with at(120.0):
            results = [limiter.allow("10.0.0.1") for _ in range(4)]
        self.assertEqual(results, [True, True, True, False])

    def test_clients_are_counted_separately(self):
        limiter = self.make_limiter(1)
        with at(120.0):
            self.assertTrue(limiter.allow("10.0.0.1"))
            self.assertTrue(limiter.allow("10.0.0.2"))
            self.assertFalse(limiter.allow("10.0.0.1"))

    def test_next_minute_starts_a_new_count(self):
        limiter = self.make_limiter(1)
        with at(120.0):
            self.assertTrue(limiter.allow("10.0.0.1"))
            self.assertFalse(limiter.allow("10.0.0.1"))
        with at(180.0):
            self.assertTrue(limiter.allow("10.0.0.1"))

    def test_zero_disables_the_limit(self):
        limiter = self.make_limiter(0)
        self.assertTrue(all(limiter.allow("10.0.0.1") for _ in range(500)))


class MemoryRateLimiterTests(RateLimiterContract, unittest.TestCase):
    def make_limiter(self, limit):
        return RateLimiter(limit)


class RedisRateLimiterTests(RateLimiterContract, unittest.TestCase):
    def make_limiter(self, limit):
        self.redis = FakeRedis()
        return RateLimiter(limit, redis=self.redis)

    def test_counter_key_expires_with_its_window(self):
        limiter = self.make_limiter(5)
        with at(120.0):
            limiter.allow("10.0.0.1")
        self.assertEqual(self.redis.ttl, {"imichat:ratelimit:10.0.0.1:2": 60})


class RetryAfterTests(unittest.TestCase):
    def test_seconds_until_the_window_ends(self):
        with at(125.0):
            self.assertEqual(RateLimiter(10).retry_after(), 55)


if __name__ == "__main__":
    unittest.main()
