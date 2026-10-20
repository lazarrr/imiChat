"""
Tests for the chat session stores (src/api/sessions.py).

    python -m unittest tests.test_sessions -v
"""

import unittest
from unittest.mock import patch

from src.api.sessions import MemorySessionStore, RedisSessionStore, connect_redis

from .fake_redis import FakeRedis

TURN = [
    {"role": "user", "content": "Koji su moduli na informatici?"},
    {"role": "assistant", "content": "Računarske nauke i Softversko inženjerstvo."},
]


class SessionStoreContract:
    """Behaviour both stores share; subclasses provide make_store()."""

    def test_new_session_has_no_history(self):
        self.assertEqual(self.make_store().get_history("s1"), [])

    def test_saved_turn_is_returned_for_its_session_only(self):
        store = self.make_store()

        store.save_turn("s1", TURN[0]["content"], TURN[1]["content"])

        self.assertEqual(store.get_history("s1"), TURN)
        self.assertEqual(store.get_history("s2"), [])
        self.assertEqual(store.active_count(), 1)

    def test_request_without_session_id_is_not_stored(self):
        store = self.make_store()

        store.save_turn(None, "pitanje", "odgovor")

        self.assertEqual(store.get_history(None), [])
        self.assertEqual(store.active_count(), 0)


class MemorySessionStoreTests(SessionStoreContract, unittest.TestCase):
    def make_store(self):
        return MemorySessionStore(ttl=60)

    def test_idle_session_expires(self):
        store = self.make_store()
        with patch("src.api.sessions.time.time", return_value=1000.0):
            store.save_turn("s1", "pitanje", "odgovor")

        with patch("src.api.sessions.time.time", return_value=1061.0):
            self.assertEqual(store.get_history("s1"), [])
            self.assertEqual(store.active_count(), 0)


class RedisSessionStoreTests(SessionStoreContract, unittest.TestCase):
    def make_store(self):
        self.redis = FakeRedis()
        return RedisSessionStore(self.redis, ttl=60)

    def test_saving_and_reading_refresh_the_expiry(self):
        store = self.make_store()

        store.save_turn("s1", "pitanje", "odgovor")
        self.assertEqual(self.redis.ttl, {"imichat:session:s1": 60})

        self.redis.ttl.clear()
        store.get_history("s1")
        self.assertEqual(self.redis.ttl, {"imichat:session:s1": 60})


class ConnectRedisTests(unittest.TestCase):
    def test_no_url_means_in_memory(self):
        self.assertIsNone(connect_redis(None))
        self.assertIsNone(connect_redis(""))

    def test_unreachable_redis_falls_back_to_memory(self):
        # Port 1 is closed (or the redis package is missing) → None + warning
        with self.assertLogs("src.api.sessions", level="WARNING"):
            self.assertIsNone(connect_redis("redis://127.0.0.1:1/0"))


if __name__ == "__main__":
    unittest.main()
