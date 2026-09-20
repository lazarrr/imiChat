# src/core/llm_manager.py
"""
LLM Manager — talks to either a local Ollama server or a remote
OpenAI-compatible server (``/v1/chat/completions``).

Backend selection is driven by the ``LLM_USE_LOCAL`` setting:
  * ``True``  → official ``ollama`` Python client (local inference)
  * ``False`` → remote OpenAI-compatible HTTP API (``requests``)

Both expose the same two interfaces:
  1.  chat()      → multi-turn conversation
  2.  generate()  → fast single-turn (e.g. query-rewrite)

Both explicitly set num_ctx so the context window is never silently truncated.
Streaming is token-by-token from either backend.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from threading import Lock
from typing import Generator

import ollama
import requests

logger = logging.getLogger(__name__)

# Deterministic sampling for query rewrites: identical (history, question)
# pairs must always produce the same standalone query.  The cache below
# guarantees it even if the LLM itself is not bit-for-bit reproducible.
_REWRITE_SEED = 42
_REWRITE_CACHE_SIZE = 256


class _OpenAICompatibleClient:
    """
    Minimal OpenAI-compatible client (``POST {base_url}/chat/completions``).

    Exposes the subset of the ``ollama.Client`` interface that
    :class:`LLMManager` relies on — ``chat`` / ``generate``, streaming or
    not — and maps Ollama-style options (``num_predict``, ``seed``) onto
    their OpenAI-compatible counterparts (``max_tokens``, ``seed``).
    """

    def __init__(self, base_url: str, api_key: str = "", timeout: float = 300.0):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = requests.Session()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict | None,
        stream: bool,
    ) -> dict:
        options = options or {}
        payload: dict = {"model": model, "messages": messages, "stream": stream}
        if options.get("temperature") is not None:
            payload["temperature"] = options["temperature"]
        if options.get("num_predict") is not None:
            payload["max_tokens"] = options["num_predict"]
        if options.get("seed") is not None:
            payload["seed"] = options["seed"]
        if options.get("repeat_penalty") is not None:
            # Ollama's OpenAI layer maps frequency_penalty to
            # repeat_penalty = 1 + frequency_penalty.
            payload["frequency_penalty"] = min(
                2.0, max(0.0, float(options["repeat_penalty"]) - 1.0)
            )
        return payload

    def _post(self, payload: dict) -> requests.Response:
        url = f"{self.base_url}/chat/completions"
        response = self._session.post(
            url,
            json=payload,
            headers=self._headers(),
            stream=bool(payload.get("stream")),
            timeout=(10, self.timeout),
        )
        if response.status_code >= 400:
            logger.error(
                "Remote LLM request to %s failed with HTTP %s: %s",
                url,
                response.status_code,
                response.text[:500],
            )
            response.raise_for_status()
        return response

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict | None = None,
        stream: bool = False,
    ):
        response = self._post(self._payload(model, messages, options, stream))
        if stream:
            return self._iter_stream(response)
        content = _OpenAICompatibleClient._json(response)["choices"][0]["message"]["content"]
        return {"message": {"role": "assistant", "content": content}}

    def generate(
        self,
        model: str,
        prompt: str,
        options: dict | None = None,
        stream: bool = False,
    ):
        messages = [{"role": "user", "content": prompt}]
        if stream:
            response = self._post(self._payload(model, messages, options, True))
            return self._iter_generate_stream(response)
        result = self.chat(model, messages, options, stream=False)
        return {"response": result["message"]["content"]}

    @staticmethod
    def _iter_stream(response: requests.Response):
        for token in _OpenAICompatibleClient._iter_tokens(response):
            yield {"message": {"content": token}}

    @staticmethod
    def _iter_generate_stream(response: requests.Response):
        for token in _OpenAICompatibleClient._iter_tokens(response):
            yield {"response": token}

    @staticmethod
    def _json(response: requests.Response) -> dict:
        # Decode explicitly as UTF-8: the server may omit the charset from
        # Content-Type, in which case requests falls back to ISO-8859-1 and
        # mangles Serbian diacritics (č, ć, š, ž, đ).
        return json.loads(response.content.decode("utf-8"))

    @staticmethod
    def _iter_tokens(response: requests.Response):
        # Same charset caveat as _json: SSE responses (text/event-stream)
        # carry no charset, so decode the raw bytes as UTF-8 ourselves
        # instead of letting requests guess ISO-8859-1.
        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line.startswith("data:"):
                line = line[len("data:"):].strip()
            if line == "[DONE]":
                break
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Ignoring non-JSON stream chunk: %r", line)
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            token = (choices[0].get("delta") or {}).get("content")
            if token:
                yield token


class LLMManager:
    # Dedicated pool so a slow rewrite can be abandoned on timeout without
    # blocking the request.  The in-flight HTTP call finishes in the
    # background; its result is simply discarded.
    _REWRITE_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="query-rewrite")

    def __init__(self, config):
        self.config = config
        self.temperature = config.LLM_TEMPERATURE
        self.num_predict = config.LLM_MAX_TOKENS
        self.num_ctx     = getattr(config, "LLM_NUM_CTX", 8192)
        self.repeat_penalty = float(getattr(config, "LLM_REPEAT_PENALTY", 1.2))
        self.repeat_last_n  = int(getattr(config, "LLM_REPEAT_LAST_N", 256))
        self.use_local   = bool(getattr(config, "LLM_USE_LOCAL", False))

        if self.use_local:
            self.model = config.LLM_MODEL
            self.base_url = getattr(
                config, "LLM_OLLAMA_BASE_URL", "http://localhost:11434"
            )
            # Persistent client so the connection is reused
            self._client = ollama.Client(host=self.base_url)
        else:
            self.model = getattr(config, "LLM_REMOTE_MODEL", "") or config.LLM_MODEL
            self.base_url = getattr(config, "LLM_REMOTE_BASE_URL", "")
            self._client = _OpenAICompatibleClient(
                base_url=self.base_url,
                api_key=getattr(config, "LLM_REMOTE_API_KEY", ""),
                timeout=float(getattr(config, "LLM_REMOTE_TIMEOUT", 300.0)),
            )

        logger.info(
            "LLM backend: %s (%s @ %s)",
            "local Ollama" if self.use_local else "remote OpenAI-compatible",
            self.model,
            self.base_url,
        )

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
            "repeat_penalty": self.repeat_penalty,
            "repeat_last_n": self.repeat_last_n,
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
        model's most recent answer.

        Uses a short /api/generate call with a small token budget so it's
        near-instant once the model is warm.

        Only the previous *answer* is given to the model (``history_text``):
        it names the entities a referential follow-up ("what can you tell me
        about them?") points to.  The prompt demands that the rewritten
        question names its topic explicitly, otherwise it is not standalone.

        Rewrites are deterministic: temperature is 0, a fixed seed is set,
        and the successful result is cached by (context, question, unclear),
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
                "2. Prepisano pitanje mora jasno da imenuje temu o kojoj se "
                "govori (npr. naziv programa, smera, modula ili predmeta), "
                "koristeci pojmove iz prethodnog ODGOVORA i najnovijeg "
                "pitanja — bez toga pitanje nije samostalno.\n"
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
                "temu najnovijeg pitanja.\n"
                "Vrati iskljucivo prepisano pitanje, bez objasnjenja.\n\n"
            )

        prompt = (
            "Tvoj zadatak je da prepises najnovije korisnicko pitanje tako da "
            "ono postane potpuno samostalno — tako da neko ko ne vidi istoriju "
            "razgovora moze da razume na sta se pitanje odnosi.\n\n"
            "Prethodni odgovor asistenta:\n"
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
