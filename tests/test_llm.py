"""Unit tests for the LLM manager query-rewrite helper (cache, seed)."""

from src.core.llm_manager import LLMManager


class _Config:
    LLM_MODEL = "test-model"
    LLM_TEMPERATURE = 0.0
    LLM_MAX_TOKENS = 64
    LLM_NUM_CTX = 2048
    LLM_OLLAMA_BASE_URL = "http://localhost:11434"
    REWRITE_SEED = 7


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
