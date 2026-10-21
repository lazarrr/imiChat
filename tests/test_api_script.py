"""
The /chat/stream route must forward the requested output script to the QA
chain (and fall back to DEFAULT_SCRIPT), instead of silently ignoring it.

    python -m unittest tests.test_api_script -v
"""

import asyncio
import json
import unittest
from unittest.mock import patch

from src.api import main as api_main


def sse_tokens(chunks):
    """Payloads of the ``data:`` lines produced by _stream_response()."""
    payloads = []
    for chunk in chunks:
        line = chunk.strip()
        if line.startswith("data:"):
            payloads.append(json.loads(line[len("data:"):].strip()))
    return payloads


class FakeQASystem:
    def __init__(self):
        self.script = None
        self.calls = 0

    async def query_stream(self, question, history, trace, script="latinic"):
        self.calls += 1
        self.script = script
        trace["script"] = script
        yield "Odgovor."


class ChatScriptTests(unittest.TestCase):
    def run_stream(self, script=None):
        fake = FakeQASystem()
        with patch.object(api_main, "qa_system", fake):
            async def collect():
                return sse_tokens([
                    chunk async for chunk in api_main._stream_response(
                        "Pitanje", None, None, script or "latinic",
                    )
                ])
            payloads = asyncio.run(collect())
        return fake, payloads

    def test_requested_cyrillic_script_reaches_the_chain(self):
        fake, payloads = self.run_stream("cyrillic")

        self.assertEqual(fake.script, "cyrillic")
        self.assertEqual(payloads[0], {"token": "Odgovor."})
        self.assertEqual(payloads[-1], {"done": True})

    def test_latin_script_is_the_default(self):
        fake, _ = self.run_stream()

        self.assertEqual(fake.script, "latinic")


class ChatRequestTests(unittest.TestCase):
    def test_script_is_optional(self):
        request = api_main.ChatRequest(question="Pitanje")

        self.assertIsNone(request.script)


if __name__ == "__main__":
    unittest.main()
