"""
Tests for the LLM wrapper (src/core/llm_manager.py).  Both backends are
mocked, so neither a local Ollama server nor the remote API is needed.

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
    """Local Ollama manager (the ollama client is mocked)."""
    config = SimpleNamespace(**{
        "LLM_MODEL": "llama3.1:latest", "LLM_TEMPERATURE": 0.0,
        "LLM_MAX_TOKENS": 2048, "LLM_NUM_CTX": 8192,
        "LLM_USE_LOCAL": True,
        **settings,
    })
    with patch.object(llm_manager.ollama, "Client") as client_class:
        manager = llm_manager.LLMManager(config)
    return manager, client_class.return_value


def make_remote_manager(**settings):
    """Remote OpenAI-compatible manager; only a requests.Session is created."""
    config = SimpleNamespace(**{
        "LLM_MODEL": "llama3.1:latest", "LLM_TEMPERATURE": 0.0,
        "LLM_MAX_TOKENS": 2048, "LLM_NUM_CTX": 8192,
        "LLM_USE_LOCAL": False,
        "LLM_REMOTE_BASE_URL": "http://llm.example:11500/v1",
        "LLM_REMOTE_MODEL": "llama3.1:8b",
        "LLM_REMOTE_API_KEY": "secret-token",
        "LLM_REMOTE_TIMEOUT": 300.0,
        **settings,
    })
    return llm_manager.LLMManager(config)


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, content=b"", status_code=200, lines=()):
        self.content = content
        self.status_code = status_code
        self.text = content.decode("utf-8", errors="replace")
        self._lines = list(lines)

    def iter_lines(self, decode_unicode=False):
        yield from self._lines

    def raise_for_status(self):
        if self.status_code >= 400:
            raise llm_manager.requests.HTTPError(f"HTTP {self.status_code}")

    def close(self):
        pass


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


@unittest.skipIf(llm_manager is None, "ollama not installed")
class ThinkingTests(unittest.TestCase):
    """Qwen3.5 thinks by default; every Ollama call must switch it off."""

    def test_every_local_call_disables_thinking(self):
        manager, client = make_manager(LLM_THINKING=False)
        client.generate.side_effect = lambda **kwargs: (
            iter([{"response": "ok"}]) if kwargs["stream"] else {"response": "ok"}
        )
        client.chat.side_effect = lambda **kwargs: (
            iter([ANSWER]) if kwargs["stream"] else ANSWER
        )
        messages = [{"role": "user", "content": "Pitanje"}]

        manager.generate("Pitanje")
        list(manager.generate_stream("Pitanje"))
        manager.chat(messages)
        list(manager.chat_stream(messages))
        manager.rewrite_query(
            "Korisnik: Koliko ESPB ima Uvod u programiranje?",
            "Koji su preduslovi za njega?",
        )

        calls = client.generate.call_args_list + client.chat.call_args_list
        self.assertEqual(len(calls), 5)
        self.assertEqual({call.kwargs["think"] for call in calls}, {False})

    def test_thinking_can_be_enabled(self):
        manager, client = make_manager(LLM_THINKING=True)
        client.generate.return_value = {"response": "ok"}

        manager.generate("Pitanje")

        self.assertIs(client.generate.call_args.kwargs["think"], True)

    def test_remote_backend_does_not_send_a_think_field(self):
        # `think` is an Ollama-only request field; the remote
        # OpenAI-compatible endpoint does not accept it.
        body = b'{"choices":[{"message":{"content":"ok"}}]}'
        with patch.object(llm_manager.requests.Session, "post",
                          return_value=FakeResponse(body)) as post:
            manager = make_remote_manager(LLM_THINKING=False)
            manager.chat([{"role": "user", "content": "Pitanje"}])

        self.assertNotIn("think", post.call_args.kwargs["json"])


@unittest.skipIf(llm_manager is None, "ollama not installed")
class SettingsDefaultsTests(unittest.TestCase):
    def test_defaults_target_local_qwen_without_thinking(self):
        from config.settings import Settings

        settings = Settings(_env_file=None)

        self.assertEqual(settings.LLM_MODEL, "qwen3.5:9b")
        self.assertTrue(settings.LLM_USE_LOCAL)
        self.assertFalse(settings.LLM_THINKING)
        self.assertEqual(settings.DEFAULT_SCRIPT, "latinic")


@unittest.skipIf(llm_manager is None, "ollama not installed")
class RewritePromptTests(unittest.TestCase):
    HISTORY = (
        "Korisnik: Koliko ESPB ima Uvod u programiranje?\n"
        "Asistent: Uvod u programiranje nosi 6 ESPB bodova."
    )
    QUESTION = "Koji su preduslovi za njega?"
    REWRITTEN = "Koji su preduslovi za Uvod u programiranje?"

    def test_rewrite_prompt_is_the_rag_rewrite_module(self):
        manager, client = make_manager()
        client.generate.return_value = {"response": self.REWRITTEN}

        manager.rewrite_query(self.HISTORY, self.QUESTION)

        prompt = client.generate.call_args.kwargs["prompt"]
        # The instructions the module must follow: rewrite only, resolve
        # references from the history, preserve the intent, add no facts,
        # keep a standalone question unchanged, ignore an unrelated topic.
        self.assertIn("Ti si modul za prepisivanje korisničkih pitanja", prompt)
        self.assertIn("Nikada nemoj da odgovaraš na korisničko pitanje", prompt)
        self.assertIn("Sačuvaj prvobitnu nameru korisnika", prompt)
        self.assertIn("Nemoj da dodaješ činjenice", prompt)
        self.assertIn("Ako istorija razgovora nije dovoljna", prompt)
        self.assertIn("novu, nepovezanu temu", prompt)
        self.assertIn("Napiši samo prepisano pitanje", prompt)
        self.assertIn("Primer 6", prompt)
        # The history and the new question are injected under the headings
        # the prompt itself uses in its examples.
        self.assertIn(f"Istorija razgovora:\n{self.HISTORY}", prompt)
        self.assertIn(f"Novo pitanje korisnika:\n{self.QUESTION}", prompt)
        # The task closes the prompt, after the data it applies to.
        self.assertTrue(prompt.rstrip().endswith(
            "Poštuj sva navedena pravila i vrati isključivo jedno prepisano pitanje."
        ))

    def test_rewrite_prompt_and_raw_response_are_logged(self):
        manager, client = make_manager()
        client.generate.return_value = {"response": f"  {self.REWRITTEN}  "}

        result = manager.rewrite_query(self.HISTORY, self.QUESTION)

        self.assertEqual(result, self.REWRITTEN)
        self.assertEqual(manager.last_rewrite["prompt"],
                         client.generate.call_args.kwargs["prompt"])
        self.assertEqual(manager.last_rewrite["raw_response"], self.REWRITTEN)
        self.assertEqual(manager.last_rewrite["returned"], self.REWRITTEN)

    def test_empty_history_returns_the_question_without_calling_the_model(self):
        manager, client = make_manager()

        result = manager.rewrite_query("   ", self.QUESTION)

        self.assertEqual(result, self.QUESTION)
        client.generate.assert_not_called()
        self.assertIsNone(manager.last_rewrite["prompt"])


@unittest.skipIf(llm_manager is None, "ollama not installed")
class RemoteBackendTests(unittest.TestCase):
    """LLM_USE_LOCAL=False sends the right OpenAI-compatible requests."""

    def test_remote_backend_is_selected_and_uses_its_own_model(self):
        manager = make_remote_manager()

        self.assertEqual(manager.backend, "remote")
        self.assertEqual(manager.model, "llama3.1:8b")
        self.assertEqual(manager.base_url, "http://llm.example:11500/v1")
        self.assertIsInstance(manager._client, llm_manager._OpenAICompatibleClient)

    def test_chat_posts_openai_payload_and_bearer_token(self):
        body = (
            '{"choices":[{"message":{"role":"assistant",'
            '"content":"Zdravo, čoveče!"}}]}'
        ).encode("utf-8")
        with patch.object(llm_manager.requests.Session, "post",
                          return_value=FakeResponse(body)) as post:
            manager = make_remote_manager()
            answer = manager.chat([{"role": "user", "content": "Zdravo"}])

        # Serbian diacritics survive the missing charset in Content-Type.
        self.assertEqual(answer, "Zdravo, čoveče!")
        args, kwargs = post.call_args
        self.assertEqual(args[0], "http://llm.example:11500/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret-token")
        self.assertFalse(kwargs["stream"])
        payload = kwargs["json"]
        self.assertEqual(payload["model"], "llama3.1:8b")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "Zdravo"}])
        self.assertEqual(payload["max_tokens"], 2048)
        self.assertEqual(payload["temperature"], 0.0)
        # Ollama repeat_penalty 1.2 → OpenAI frequency_penalty 0.2.
        self.assertAlmostEqual(payload["frequency_penalty"], 0.2)

    def test_chat_stream_parses_sse_deltas(self):
        lines = [
            b'data: {"choices":[{"delta":{"content":"Zdra"}}]}',
            b"",
            'data: {"choices":[{"delta":{"content":"vo, č"}}]}'.encode("utf-8"),
            b"data: [DONE]",
        ]
        with patch.object(llm_manager.requests.Session, "post",
                          return_value=FakeResponse(lines=lines)) as post:
            manager = make_remote_manager()
            tokens = list(manager.chat_stream([{"role": "user", "content": "Hi"}]))

        self.assertEqual(tokens, ["Zdra", "vo, č"])
        self.assertTrue(post.call_args.kwargs["stream"])
        self.assertTrue(post.call_args.kwargs["json"]["stream"])

    def test_query_rewrite_uses_the_remote_endpoint(self):
        question = "Koji su preduslovi za njega?"
        history = "Korisnik: Koliko ESPB ima Uvod u programiranje?"
        rewritten = "Koji su preduslovi za Uvod u programiranje?"
        body = f'{{"choices":[{{"message":{{"content":"{rewritten}"}}}}]}}'.encode()
        with patch.object(llm_manager.requests.Session, "post",
                          return_value=FakeResponse(body)) as post:
            manager = make_remote_manager()
            result = manager.rewrite_query(history, question)

        self.assertEqual(result, rewritten)
        self.assertEqual(manager.last_rewrite["raw_response"], rewritten)
        self.assertEqual(manager.last_rewrite["returned"], rewritten)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["max_tokens"], 128)
        self.assertIn(history, payload["messages"][0]["content"])
        self.assertIn(question, payload["messages"][0]["content"])

    def test_http_error_is_raised(self):
        with patch.object(llm_manager.requests.Session, "post",
                          return_value=FakeResponse(b'{"error":"boom"}', 500)):
            manager = make_remote_manager()
            with self.assertRaises(llm_manager.requests.HTTPError):
                manager.chat([{"role": "user", "content": "Hi"}])


if __name__ == "__main__":
    unittest.main()
