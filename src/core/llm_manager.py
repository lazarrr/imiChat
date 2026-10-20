# src/core/llm_manager.py
"""
LLM Manager — talks to either a local Ollama server or a remote
OpenAI-compatible server (``/v1/chat/completions``).

Backend selection is driven by the ``LLM_USE_LOCAL`` setting:
  * ``True``  → official ``ollama`` Python client (local inference)
  * ``False`` → remote OpenAI-compatible HTTP API (``requests``), the default

Both expose the same two interfaces:
  1.  chat()      → multi-turn conversation
  2.  generate()  → fast single-turn (e.g. query-rewrite)

Both explicitly set the context window where the backend supports it.
Streaming is token-by-token from either backend.

Local Ollama calls always pass ``think`` from ``LLM_THINKING`` (off by
default): reasoning models such as Qwen3.5 otherwise return a long thinking
block before the answer, which the chatbot does not use.
"""

from __future__ import annotations

import json
import logging
from typing import Generator

import ollama
import requests

logger = logging.getLogger(__name__)


class _OpenAICompatibleClient:
    """
    Minimal OpenAI-compatible client (``POST {base_url}/chat/completions``).

    Exposes the subset of the ``ollama.Client`` interface that
    :class:`LLMManager` relies on — ``chat`` / ``generate``, streaming or
    not — and maps Ollama-style options (``num_predict``, ``repeat_penalty``)
    onto their OpenAI-compatible counterparts (``max_tokens``,
    ``frequency_penalty``).
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
        data = self._json(response)
        content = data["choices"][0]["message"].get("content") or ""
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
        try:
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
                choice = choices[0]
                token = (choice.get("delta") or {}).get("content")
                if token is None:
                    # A server may ignore stream=True and return one full
                    # completion instead of SSE deltas.
                    token = (choice.get("message") or {}).get("content")
                if token:
                    yield token
        finally:
            # Release the connection when the consumer stops early, e.g. the
            # browser disconnects while the answer is still streaming.
            response.close()


# ── Query-rewrite prompt ─────────────────────────────────────────────── #
# A dedicated rewrite module for the RAG retriever: it turns the latest
# question into a standalone one, resolving pronouns and short follow-ups
# from the last 1–2 turns, and never answers.  The prompt is in Serbian
# (Latin script) because the corpus and the conversations are.
#
# Placeholders:
#   {history_text}     the last turns as "Korisnik:"/"Asistent:" lines
#   {latest_question}  the latest question exactly as the user typed it
QUERY_REWRITE_PROMPT = """\
Ti si modul za prepisivanje korisničkih pitanja za RAG sistem koji pretražuje
vektorsku bazu podataka o fakultetu, studijskim programima, modulima,
predmetima, profesorima, upisu i studentskim aktivnostima.

TVOJ ZADATAK:
Na osnovu istorije razgovora i poslednjeg korisničkog pitanja, prepiši
poslednje pitanje tako da bude potpuno razumljivo i bez prethodnog konteksta,
kako bi moglo direktno da se prosledi sistemu za semantičku pretragu.

PRAVILA:
1. Nikada nemoj da odgovaraš na korisničko pitanje. Samo ga prepiši.
2. Koristi istoriju razgovora da razumeš na šta se odnose nejasne reči,
   zamenice i kratka dopunska pitanja, kao što su:
   "a oni?", "a koji su?", "a gde?", "koliko traje?", "može malo više?"
   i "a šta je sa tim?".
3. U prepisano pitanje uključi konkretan naziv teme, studijskog programa,
   modula, predmeta, profesora ili druge relevantne oblasti ako se taj podatak
   može pouzdano utvrditi iz istorije razgovora.
4. Sačuvaj prvobitnu nameru korisnika. Nemoj da menjaš značenje pitanja.
5. Nemoj da dodaješ činjenice, pretpostavke ili detalje koji nisu potkrepljeni
   korisničkim pitanjem ili istorijom razgovora.
6. Ako je poslednje pitanje već jasno i samostalno, zadrži ga bez nepotrebnih
   izmena.
7. Ako istorija razgovora nije dovoljna da pouzdano razumeš pitanje, vrati
   poslednje korisničko pitanje bez promene značenja.
8. Ako korisnik započne novu, nepovezanu temu, nemoj da koristiš prethodni
   razgovor da menjaš značenje novog pitanja.
9. Napiši samo prepisano pitanje. Bez odgovora, objašnjenja, uvoda, navodnika
   ili oznaka kao što su "Prepisano pitanje:".

PRIMERI:

Primer 1
Istorija razgovora:
Korisnik: Koliko modula ima na OAS Informatike?
Asistent: Ima ih tri.
Novo pitanje korisnika: A oni su?
Izlaz:
Koji su moduli na OAS Informatike?

Primer 2
Istorija razgovora:
Korisnik: Koji su moduli na OAS Informatike?
Asistent: Jedan od modula je Softversko inženjerstvo i informacione tehnologije.
Novo pitanje korisnika: Možeš li da mi kažeš nešto više o njemu?
Izlaz:
Koje su informacije o modulu Softversko inženjerstvo i informacione tehnologije
na OAS Informatike?

Primer 3
Istorija razgovora:
Korisnik: Kako se prijavljuje ispit?
Asistent: Ispit se prijavljuje preko studentskog servisa.
Novo pitanje korisnika: A kako se odjavljuje?
Izlaz:
Kako se odjavljuje ispit preko studentskog servisa?

Primer 4
Istorija razgovora:
Korisnik: Koji su uslovi za upis na OAS Informatike?
Asistent: Potrebno je ispuniti propisane uslove za upis.
Novo pitanje korisnika: A za budžet?
Izlaz:
Koji su uslovi za upis na budžet na OAS Informatike?

Primer 5
Istorija razgovora:
Korisnik: Gde mogu da pronađem raspored ispita?
Asistent: Raspored je dostupan na sajtu fakulteta.
Novo pitanje korisnika: A za kolokvijume?
Izlaz:
Gde mogu da pronađem raspored kolokvijuma na sajtu fakulteta?

Primer 6
Istorija razgovora:
Korisnik: Koliko modula ima na OAS Informatike?
Asistent: Ima ih tri.
Novo pitanje korisnika: Kako funkcioniše mašinsko učenje?
Izlaz:
Kako funkcioniše mašinsko učenje?

---

Istorija razgovora:
{history_text}

Novo pitanje korisnika:
{latest_question}

---

Sada prepiši novo korisničko pitanje na osnovu dostavljene istorije razgovora.
Poštuj sva navedena pravila i vrati isključivo jedno prepisano pitanje.\
"""


class LLMManager:
    def __init__(self, config):
        self.config = config
        self.temperature = config.LLM_TEMPERATURE
        self.num_predict = config.LLM_MAX_TOKENS
        self.num_ctx     = getattr(config, "LLM_NUM_CTX", 8192)
        self.repeat_penalty = float(getattr(config, "LLM_REPEAT_PENALTY", 1.2))
        self.repeat_last_n  = int(getattr(config, "LLM_REPEAT_LAST_N", 256))
        self.use_local   = bool(getattr(config, "LLM_USE_LOCAL", False))
        # Off by default: the chatbot wants the answer, not the reasoning
        # trace.  Only Ollama takes a per-call `think` flag.
        self.thinking    = bool(getattr(config, "LLM_THINKING", False))

        # Prompt and raw output of the most recent rewrite_query() call, kept
        # for the request log (see src/api/request_log.py).  None until then.
        self.last_rewrite: dict | None = None

        if self.use_local:
            self.backend = "local"
            self.model = config.LLM_MODEL
            self.base_url = getattr(
                config, "LLM_OLLAMA_BASE_URL", "http://localhost:11434"
            )
            # Persistent client so the connection is reused
            self._client = ollama.Client(host=self.base_url)
        else:
            self.backend = "remote"
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

    @property
    def _thinking_options(self) -> dict:
        """
        Per-call reasoning switch for the local Ollama client.

        ``think`` is a top-level request field in Ollama, not a model option,
        so it has to be passed to every chat/generate call.  The remote
        OpenAI-compatible backend has no equivalent field, so it gets none.
        """
        return {"think": self.thinking} if self.use_local else {}

    # ------------------------------------------------------------------ #
    #  Single-turn generation  (simple prompt → completion)              #
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str) -> str:
        """Non-streaming single-turn generation (Ollama /api/generate)."""
        resp = self._client.generate(
            model=self.model,
            prompt=prompt,
            options=self._options,
            stream=False,
            **self._thinking_options,
        )
        return resp["response"]

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Token-by-token streaming single-turn generation."""
        for chunk in self._client.generate(
            model=self.model,
            prompt=prompt,
            options=self._options,
            stream=True,
            **self._thinking_options,
        ):
            token = chunk.get("response", "")
            if token:
                yield token

    # ------------------------------------------------------------------ #
    #  Chat  (messages list → completion)                                 #
    # ------------------------------------------------------------------ #

    def chat(self, messages: list[dict[str, str]]) -> str:
        """
        Non-streaming chat against the selected backend.

        messages: [{"role":"system"|"user"|"assistant", "content":"..."}, ...]
        """
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            options=self._options,
            stream=False,
            **self._thinking_options,
        )
        return resp["message"]["content"]

    def chat_stream(self, messages: list[dict[str, str]]) -> Generator[str, None, None]:
        """
        Token-by-token streaming from the selected backend.

        Each chunk is yielded immediately as it arrives: Ollama's native
        stream for a local server, SSE ``delta`` chunks for a remote
        OpenAI-compatible server.
        """
        for chunk in self._client.chat(
            model=self.model,
            messages=messages,
            options=self._options,
            stream=True,
            **self._thinking_options,
        ):
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token

    # ------------------------------------------------------------------ #
    #  Query-rewrite helper  (fast, low-token, single-turn)               #
    # ------------------------------------------------------------------ #

    def rewrite_query(self, history_text: str, latest_question: str) -> str:
        """
        Rewrite an unclear follow-up question (e.g. "give me more
        information on them") into a standalone question using the last
        1–2 turns of conversation history.

        The prompt (``QUERY_REWRITE_PROMPT``) casts the model as a dedicated
        rewrite module for the RAG retriever.  It must not answer: it
        resolves pronouns and short follow-ups from the history, names the
        topic or entity they refer to, preserves the original intent, adds
        no facts, leaves an already standalone question unchanged, ignores
        the history for an unrelated new topic, and returns only the
        rewritten question.  The answering LLM never sees the history, so
        the rewrite must spell out both the entities and the topic.

        Only called for questions that refer back to earlier context (see
        ``_UNCLEAR_PATTERNS`` in ``qa_chain.py``); a clear question goes
        straight to retrieval.

        ``self.last_rewrite`` keeps the prompt and the raw model output of
        the most recent call, so the API can log exactly what the rewrite
        was given (see src/api/request_log.py).  The return value is
        unchanged: the rewritten question, or ``latest_question`` when the
        rewrite is unusable.

        Uses a short single-turn call with a small token budget so it's
        near-instant (< 1 s typical) once the model is warm.

        The prompt is in Serbian because both the documents and the
        conversation are in Serbian (Latin script).
        """
        self.last_rewrite = {
            "prompt": None, "raw_response": None, "returned": None,
            "history_text": history_text, "latest_question": latest_question,
        }
        if not history_text.strip():
            self.last_rewrite["returned"] = latest_question
            return latest_question

        prompt = QUERY_REWRITE_PROMPT.format(
            history_text=history_text.strip(),
            latest_question=latest_question.strip(),
        )

        self.last_rewrite["prompt"] = prompt

        resp = self._client.generate(
            model=self.model,
            prompt=prompt,
            # Same num_ctx as the answer calls: Ollama reloads the model
            # whenever num_ctx changes, which would cost two reloads per
            # rewritten question (rewrite, then answer).
            options={"temperature": 0.0, "num_predict": 128, "num_ctx": self.num_ctx},
            stream=False,
            **self._thinking_options,
        )
        result = resp["response"].strip()
        self.last_rewrite["raw_response"] = result
        # Guard: if the model returned something nonsensical, fall back
        if not result or len(result) < 3:
            self.last_rewrite["returned"] = latest_question
            return latest_question
        self.last_rewrite["returned"] = result
        return result