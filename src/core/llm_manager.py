# src/core/llm_manager.py
"""
LLM Manager — wraps Ollama via the official ``ollama`` Python client.

Two interfaces:
  1.  ollama.chat()      →  /api/chat   (multi-turn conversation)
  2.  ollama.generate()  →  /api/generate  (fast single-turn, e.g. query-rewrite)

Both explicitly set num_ctx so the context window is never silently truncated.
Streaming uses the client's native ``stream=True`` — token-by-token, same as v1.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from threading import Lock
from typing import Generator

import ollama

logger = logging.getLogger(__name__)

# Deterministic sampling for query rewrites: identical (history, question)
# pairs must always produce the same standalone query.  The cache below
# guarantees it even if the LLM itself is not bit-for-bit reproducible.
_REWRITE_SEED = 42
_REWRITE_CACHE_SIZE = 256


class LLMManager:
    # Dedicated pool so a slow rewrite can be abandoned on timeout without
    # blocking the request.  The in-flight HTTP call finishes in the
    # background; its result is simply discarded.
    _REWRITE_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="query-rewrite")

    def __init__(self, config):
        self.config = config
        self.model   = config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE
        self.num_predict = config.LLM_MAX_TOKENS
        self.num_ctx     = getattr(config, "LLM_NUM_CTX", 8192)
        self.base_url = getattr(config, "LLM_OLLAMA_BASE_URL", "http://localhost:11434")

        # Create a persistent client so the connection is reused
        self._client = ollama.Client(host=self.base_url)

        self.rewrite_seed = int(getattr(config, "REWRITE_SEED", _REWRITE_SEED))
        self._rewrite_cache: OrderedDict[str, str] = OrderedDict()
        self._rewrite_cache_lock = Lock()

    # ------------------------------------------------------------------ #
    #  Rewrite cache (same history + question → same standalone query)     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rewrite_cache_key(
        history_text: str, latest_question: str, is_unclear: bool
    ) -> str:
        payload = f"{int(is_unclear)}\n{latest_question.strip()}\n{history_text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _rewrite_cache_get(self, key: str) -> str | None:
        with self._rewrite_cache_lock:
            value = self._rewrite_cache.get(key)
            if value is not None:
                self._rewrite_cache.move_to_end(key)
            return value

    def _rewrite_cache_put(self, key: str, value: str) -> None:
        with self._rewrite_cache_lock:
            self._rewrite_cache[key] = value
            self._rewrite_cache.move_to_end(key)
            while len(self._rewrite_cache) > _REWRITE_CACHE_SIZE:
                self._rewrite_cache.popitem(last=False)

    # ------------------------------------------------------------------ #
    #  Shared options dict                                                #
    # ------------------------------------------------------------------ #

    @property
    def _options(self) -> dict:
        return {
            "temperature": self.temperature,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
        }

    # ------------------------------------------------------------------ #
    #  /api/generate  (simple prompt → completion)                        #
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str) -> str:
        """Non-streaming generation via /api/generate"""
        resp = self._client.generate(
            model=self.model,
            prompt=prompt,
            options=self._options,
            stream=False,
        )
        return resp["response"]

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Token-by-token streaming via /api/generate"""
        for chunk in self._client.generate(
            model=self.model,
            prompt=prompt,
            options=self._options,
            stream=True,
        ):
            token = chunk.get("response", "")
            if token:
                yield token

    # ------------------------------------------------------------------ #
    #  /api/chat  (messages list → completion)                            #
    # ------------------------------------------------------------------ #

    def chat(self, messages: list[dict[str, str]]) -> str:
        """
        Non-streaming chat via Ollama /api/chat.

        messages: [{"role":"system"|"user"|"assistant", "content":"..."}, ...]
        """
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            options=self._options,
            stream=False,
        )
        return resp["message"]["content"]

    def chat_stream(self, messages: list[dict[str, str]]) -> Generator[str, None, None]:
        """
        Token-by-token streaming via Ollama /api/chat.

        Uses the official client's native ``stream=True`` — each chunk
        is yielded immediately as it arrives from the Ollama server.
        """
        for chunk in self._client.chat(
            model=self.model,
            messages=messages,
            options=self._options,
            stream=True,
        ):
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token

    # ------------------------------------------------------------------ #
    #  Query-rewrite helper  (fast, low-token, single-turn)               #
    # ------------------------------------------------------------------ #

    def rewrite_query(
        self,
        history_text: str,
        latest_question: str,
        is_unclear: bool = False,
        timeout: float | None = None,
    ) -> str:
        """
        Rewrite a follow-up question into a standalone question using the
        last 1–2 turns of conversation history.

        Uses a short /api/generate call with a small token budget so it's
        near-instant once the model is warm.

        Both the previous *questions* and the previous *answers* are given
        to the model: the answer usually names the entities a follow-up
        ("what can you tell me about them?") refers to, while the previous
        question anchors the topic (e.g. "OAS informatika").  The prompt
        explicitly demands that the previous question's topic is preserved,
        otherwise the rewrite drifts to a different subject.

        Rewrites are deterministic: temperature is 0, a fixed seed is set,
        and the successful result is cached by (history, question, unclear),
        so the same conversation state always yields the same query.

        ``timeout`` caps how long the caller waits; on timeout the original
        question is returned so the QA chain can apply its own fallback.

        The prompt is in Serbian because both the documents and the
        conversation are in Serbian (Latin script).
        """
        if not history_text.strip():
            return latest_question

        if is_unclear:
            reference_rule = (
                "Najnovije pitanje je NEJASNO i oslanja se na prethodni "
                "razgovor (na primer: 'daj mi vise informacija o njima', "
                "'sta je sa tim', 'pojasni').\n"
                "1. Zameni nejasne reci (npr. 'njima', 'to', 'tim') konkretnim "
                "pojmovima iz prethodnog ODGOVORA asistenta (nazivi smerova, "
                "modula, predmeta, programa).\n"
                "2. Temu iz prethodnog PITANJA korisnika (npr. naziv programa "
                "ili smera) obavezno ukljuci u prepisano pitanje — bez nje "
                "pitanje nije samostalno.\n"
                "3. Ako najnovije pitanje trazi DODATNE informacije o necemu "
                "iz prethodnog odgovora, prepisi ga tako da jasno imenuje "
                "temu o kojoj se traze dodatne informacije, ali NE dodaj "
                "nove cinjenice.\n"
                "4. Ne izmisljaj pojmove kojih nema u istoriji razgovora.\n"
                "5. Vrati iskljucivo prepisano pitanje, bez objasnjenja.\n\n"
            )
        else:
            reference_rule = (
                "Ako se najnovije pitanje odnosi na nesto iz prethodnog "
                "ODGOVORA asistenta, razresi to pomocu tog odgovora i sacuvaj "
                "temu najnovijeg pitanja. NE spajaj prethodno pitanje sa "
                "najnovijim — prethodno pitanje sluzi samo kao kontekst.\n"
                "Vrati iskljucivo prepisano pitanje, bez objasnjenja.\n\n"
            )

        prompt = (
            "Tvoj zadatak je da prepises najnovije korisnicko pitanje tako da "
            "ono postane potpuno samostalno — tako da neko ko ne vidi istoriju "
            "razgovora moze da razume na sta se pitanje odnosi.\n\n"
            "Istorija razgovora (sadrzi i odgovore asistenta):\n"
            f"{history_text}\n\n"
            "Najnovije pitanje: "
            f"{latest_question}\n\n"
            "Pravila:\n"
            + reference_rule)

        cache_key = self._rewrite_cache_key(
            history_text, latest_question, is_unclear
        )
        cached = self._rewrite_cache_get(cache_key)
        if cached is not None:
            logger.info(
                "Query rewrite cache hit: %r → %r", latest_question, cached
            )
            return cached

        def _generate() -> str:
            resp = self._client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.0,
                    "num_predict": 128,
                    "num_ctx": 2048,
                    "seed": self.rewrite_seed,
                },
                stream=False,
            )
            return resp["response"]

        if timeout is None:
            result = _generate()
        else:
            future = self._REWRITE_POOL.submit(_generate)
            try:
                result = future.result(timeout=timeout)
            except FutureTimeoutError:
                logger.warning(
                    "Query rewrite timed out after %.1fs — using original question",
                    timeout,
                )
                future.cancel()
                return latest_question
            except Exception:
                logger.exception("Query rewrite failed — using original question")
                return latest_question

        result = (result or "").strip()
        # Guard: if the model returned something nonsensical, fall back
        if not result or len(result) < 3:
            return latest_question
        self._rewrite_cache_put(cache_key, result)
        return result
