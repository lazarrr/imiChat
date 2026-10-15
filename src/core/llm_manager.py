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

from typing import Generator

import ollama


class LLMManager:
    def __init__(self, config):
        self.config = config
        self.model   = config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE
        self.num_predict = config.LLM_MAX_TOKENS
        self.num_ctx     = getattr(config, "LLM_NUM_CTX", 8192)
        self.base_url = getattr(config, "LLM_OLLAMA_BASE_URL", "http://localhost:11434")

        # Create a persistent client so the connection is reused
        self._client = ollama.Client(host=self.base_url)

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

    def rewrite_query(self, history_text: str, latest_question: str) -> str:
        """
        Rewrite an unclear follow-up question (e.g. "give me more
        information on them") into a standalone question using the last
        1–2 turns of conversation history.

        Only called for questions that refer back to earlier context.  The
        previous assistant answer is the main reference point for resolving
        them; the previous user question is used only when the answer does
        not name what the question refers to.  The answering LLM never sees
        the history, so the rewrite must name the referenced entities.

        Uses a short /api/generate call with a small token budget so it's
        near-instant (< 1 s typical).

        The prompt is in Serbian because both the documents and the
        conversation are in Serbian (Latin script).
        """
        if not history_text.strip():
            return latest_question

        prompt = (
            "Tvoj zadatak je da prepises najnovije korisnicko pitanje tako da "
            "ono postane potpuno samostalno — tako da neko ko ne vidi istoriju "
            "razgovora moze da razume na sta se pitanje odnosi.\n\n"
            "Istorija razgovora (sadrzi i odgovore asistenta):\n"
            f"{history_text}\n\n"
            "Najnovije pitanje: "
            f"{latest_question}\n\n"
            "Pravila:\n"
            "Najnovije pitanje se verovatno oslanja na kontekst prethodnog "
            "ODGOVORA asistenta (na primer: 'daj mi vise informacija o njima', "
            "'sta je sa tim', 'pojasni'). Da bi razresio na sta se pitanje "
            "odnosi, koristi prethodni ODGOVOR asistenta; prethodno pitanje "
            "korisnika koristi samo ako odgovor ne imenuje pojam na koji se "
            "pitanje odnosi. Zameni nejasne reci konkretnim pojmovima (npr. "
            "nazivi modula, predmeta, programa). NE spajaj prethodno pitanje "
            "sa najnovijim — prethodno pitanje sluzi samo kao kontekst.\n"
            "Ako je najnovije pitanje vec razumljivo samo po sebi, vrati ga "
            "nepromenjeno.\n\n"
            "Ne odgovaraj na pitanje. Vrati SAMO prepisano pitanje, nista drugo. "
            "Ne dodaj navodnike, objasnjenja ni komentare. Prepisi ga na srpskom "
            "latinskom pismu."
        )

        resp = self._client.generate(
            model=self.model,
            prompt=prompt,
            # Same num_ctx as the answer calls: Ollama reloads the model
            # whenever num_ctx changes, which would cost two reloads per
            # rewritten question (rewrite, then answer).
            options={"temperature": 0.0, "num_predict": 128, "num_ctx": self.num_ctx},
            stream=False,
        )
        result = resp["response"].strip()
        # Guard: if the model returned something nonsensical, fall back
        if not result or len(result) < 3:
            return latest_question
        return result