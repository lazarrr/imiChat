"""
Tests for the Ollama wrapper (src/core/llm_manager.py).  The Ollama client
is a mock, so no Ollama server is needed.

    python -m unittest tests.test_llm -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from src.core import llm_manager
except ImportError:  # the ollama package is not installed
    llm_manager = None

ANSWER = {"message": {"content": "Odgovor."}}


def make_manager(**settings):
    config = SimpleNamespace(**{
        "LLM_MODEL": "llama3.1:latest", "LLM_TEMPERATURE": 0.0,
        "LLM_MAX_TOKENS": 2048, "LLM_NUM_CTX": 8192,
        **settings,
    })
    with patch.object(llm_manager.ollama, "Client") as client_class:
        manager = llm_manager.LLMManager(config)
    return manager, client_class.return_value


@unittest.skipIf(llm_manager is None, "ollama not installed")
class ContextWindowTests(unittest.TestCase):
    def test_rewrite_and_answers_use_the_same_context_window(self):
        # Ollama reloads the model whenever num_ctx changes between calls,
        # so a rewrite with its own window would cost two reloads per
        # rewritten question (rewrite, then answer).
        manager, client = make_manager(LLM_NUM_CTX=8192)
        client.generate.return_value = {"response": "Koji su preduslovi za Uvod u programiranje?"}
        client.chat.side_effect = lambda **kwargs: iter([ANSWER]) if kwargs["stream"] else ANSWER
        messages = [{"role": "user", "content": "Pitanje"}]

        manager.rewrite_query("Korisnik: Koliko ESPB ima Uvod u programiranje?",
                              "Koji su preduslovi za njega?")
        manager.chat(messages)
        list(manager.chat_stream(messages))

        calls = client.generate.call_args_list + client.chat.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertEqual({call.kwargs["options"]["num_ctx"] for call in calls}, {8192})


if __name__ == "__main__":
    unittest.main()
