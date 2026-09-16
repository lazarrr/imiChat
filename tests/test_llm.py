"""Unit tests for the LLM manager (backend selection, remote client, rewrite)."""

import json

import pytest
import requests

from src.core.llm_manager import LLMManager


class _Config:
    LLM_MODEL = "test-model"
    LLM_USE_LOCAL = True
    LLM_TEMPERATURE = 0.0
    LLM_MAX_TOKENS = 64
    LLM_NUM_CTX = 2048
    LLM_OLLAMA_BASE_URL = "http://localhost:11434"
    LLM_REPEAT_PENALTY = 1.2
    LLM_REPEAT_LAST_N = 256
    REWRITE_SEED = 7


class _RemoteConfig(_Config):
    LLM_USE_LOCAL = False
    LLM_REMOTE_BASE_URL = "http://remote.example:11500/v1"
    LLM_REMOTE_MODEL = "llama3.1:8b"
    LLM_REMOTE_API_KEY = "token"
    LLM_REMOTE_TIMEOUT = 5.0


class _FakeResponse:
    def __init__(self, payload=None, lines=None, status_code=200):
        self._payload = payload or {}
        self._lines = lines or []
        self.status_code = status_code

    @property
    def text(self):
        return json.dumps(self._payload)

    @property
    def content(self):
        return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode=False):
        for line in self._lines:
            raw = line.encode("utf-8") if isinstance(line, str) else line
            yield raw.decode("utf-8") if decode_unicode else raw

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class _FakeSession:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def post(self, url, json=None, headers=None, stream=False, timeout=None):
        self.calls.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "stream": stream,
                "timeout": timeout,
            }
        )
        return self._response


class _FakeOllamaClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, model, prompt, options, stream):
        self.calls.append({"prompt": prompt, "options": options})
        response = self._responses.pop(0) if self._responses else "fallback"
        return {"response": response}


def _manager(responses):
    manager = LLMManager(_Config())
    manager._client = _FakeOllamaClient(responses)
    return manager


def test_rewrite_is_cached_for_identical_history_and_question():
    manager = _manager(["Koji su smerovi na OAS informatici?"])
    history = "Korisnik: Koliko ima smerova?\nAsistent: Dva."
    question = "Šta možeš o njima?"

    first = manager.rewrite_query(history, question, is_unclear=True)
    second = manager.rewrite_query(history, question, is_unclear=True)

    assert first == second == "Koji su smerovi na OAS informatici?"
    assert len(manager._client.calls) == 1


def test_rewrite_cache_distinguishes_questions():
    manager = _manager(["Prvi upit", "Drugi upit"])
    history = "Korisnik: Pitanje\nAsistent: Odgovor."

    assert manager.rewrite_query(history, "Pitanje jedan?", is_unclear=True) == "Prvi upit"
    assert manager.rewrite_query(history, "Pitanje dva?", is_unclear=True) == "Drugi upit"
    assert len(manager._client.calls) == 2


def test_rewrite_uses_fixed_seed_and_greedy_decoding():
    manager = _manager(["Prepisano pitanje?"])
    history = "Korisnik: Pitanje\nAsistent: Odgovor."

    manager.rewrite_query(history, "Šta o njima?", is_unclear=True)

    options = manager._client.calls[0]["options"]
    assert options["temperature"] == 0.0
    assert options["seed"] == 7


def test_rewrite_without_history_returns_question_unchanged():
    manager = _manager(["ne sme se pozvati"])

    assert manager.rewrite_query("   ", "Pitanje?") == "Pitanje?"
    assert manager._client.calls == []


def test_local_backend_used_when_configured():
    manager = LLMManager(_Config())

    assert manager.use_local is True
    assert manager.model == "test-model"


def test_remote_backend_used_when_local_disabled():
    manager = LLMManager(_RemoteConfig())

    assert manager.use_local is False
    assert manager.model == "llama3.1:8b"
    assert manager._client.base_url == "http://remote.example:11500/v1"


def test_remote_generate_posts_openai_payload_with_auth():
    response = _FakeResponse(payload={"choices": [{"message": {"content": "Zdravo"}}]})
    manager = LLMManager(_RemoteConfig())
    manager._client._session = _FakeSession(response)

    assert manager.generate("Pitanje") == "Zdravo"

    call = manager._client._session.calls[0]
    assert call["url"] == "http://remote.example:11500/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer token"
    assert call["json"] == {
        "model": "llama3.1:8b",
        "messages": [{"role": "user", "content": "Pitanje"}],
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 64,
        "frequency_penalty": pytest.approx(0.2),
    }


def test_local_options_include_repetition_controls():
    manager = LLMManager(_Config())

    options = manager._options

    assert options["repeat_penalty"] == 1.2
    assert options["repeat_last_n"] == 256


def test_remote_maps_repeat_penalty_to_frequency_penalty():
    manager = LLMManager(_RemoteConfig())
    manager._client._session = _FakeSession(
        _FakeResponse(payload={"choices": [{"message": {"content": "ok"}}]})
    )

    manager._client.generate(
        "m", "p", options={"repeat_penalty": 1.5}, stream=False
    )

    assert manager._client._session.calls[0]["json"]["frequency_penalty"] == 0.5


def test_remote_generate_maps_rewrite_seed():
    response = _FakeResponse(payload={"choices": [{"message": {"content": "Prepisano?"}}]})
    manager = LLMManager(_RemoteConfig())
    manager._client._session = _FakeSession(response)
    history = "Korisnik: Pitanje\nAsistent: Odgovor."

    manager.rewrite_query(history, "Šta o njima?", is_unclear=True)

    payload = manager._client._session.calls[0]["json"]
    assert payload["seed"] == 7
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 128


def test_remote_chat_stream_yields_tokens():
    lines = [
        'data: {"choices": [{"delta": {"content": "Zdra"}}]}',
        "",
        'data: {"choices": [{"delta": {"content": "vo"}}]}',
        "data: [DONE]",
    ]
    manager = LLMManager(_RemoteConfig())
    manager._client._session = _FakeSession(_FakeResponse(lines=lines))

    tokens = list(manager.chat_stream([{"role": "user", "content": "Zdravo"}]))

    assert tokens == ["Zdra", "vo"]
    assert manager._client._session.calls[0]["json"]["stream"] is True


def test_remote_http_error_raises():
    manager = LLMManager(_RemoteConfig())
    manager._client._session = _FakeSession(_FakeResponse(status_code=401))

    with pytest.raises(requests.HTTPError):
        manager.generate("Pitanje")


def test_remote_stream_decodes_utf8_diacritics():
    lines = [
        'data: {"choices": [{"delta": {"content": "čćšžđ"}}]}',
        "data: [DONE]",
    ]
    manager = LLMManager(_RemoteConfig())
    manager._client._session = _FakeSession(_FakeResponse(lines=lines))

    tokens = list(manager.chat_stream([{"role": "user", "content": "reci"}]))

    assert tokens == ["čćšžđ"]


def test_remote_non_stream_decodes_utf8_diacritics():
    response = _FakeResponse(
        payload={"choices": [{"message": {"content": "čćšžđ"}}]}
    )
    manager = LLMManager(_RemoteConfig())
    manager._client._session = _FakeSession(response)

    assert manager.chat([{"role": "user", "content": "reci"}]) == "čćšžđ"
