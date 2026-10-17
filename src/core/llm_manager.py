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

        # Prompt and raw output of the most recent rewrite_query() call, kept
        # for the request log (see src/api/request_log.py).  None until then.
        self.last_rewrite: dict | None = None

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

        Only called for questions that refer back to earlier context.  Both
        the previous user question and the previous assistant answer are
        usable: the answer names the entities a pronoun refers to, while the
        previous question carries the topic that elliptical follow-ups
        ("A koliko traje?") leave out — typically its subject is not repeated
        in the answer.  The answering LLM never sees the history, so the
        rewrite must spell out both the entities and the topic.

        ``self.last_rewrite`` keeps the prompt and the raw model output of
        the most recent call, so the API can log exactly what the rewrite
        was given (see src/api/request_log.py).  The return value is
        unchanged: the rewritten question, or ``latest_question`` when the
        rewrite is unusable.

        Uses a short /api/generate call with a small token budget so it's
        near-instant (< 1 s typical).

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

        prompt = (
            "Tvoj zadatak je da prepises najnovije korisnicko pitanje tako da "
            "ono postane potpuno samostalno — tako da neko ko ne vidi istoriju "
            "razgovora moze da razume na sta se pitanje odnosi.\n\n"
            "Istorija razgovora (sadrzi i odgovore asistenta):\n"
            f"{history_text}\n\n"
            "Najnovije pitanje: "
            f"{latest_question}\n\n"
            "Pravila:\n"
            "Najnovije pitanje se oslanja na kontekst prethodnog razgovora. "
            "Koristi OBA prethodna elementa: prethodno PITANJE korisnika nosi "
            "temu i nazive pojmova (predmet, modul, program, dokument), a "
            "prethodni ODGOVOR asistenta imenuje na sta se odnose zamenice i "
            "izrazi poput 'njima', 'to', 'taj'. "
            "Ako je najnovije pitanje elipticno — na primer 'A koliko traje?', "
            "'A kada?', 'I sta jos?' — ono nasledjuje temu iz prethodnog "
            "PITANJA, pa u prepisano pitanje ukljuci taj predmet ili pojam iz "
            "prethodnog pitanja, i onda razresi zamenice pomocu prethodnog "
            "ODGOVORA. Zameni nejasne reci konkretnim pojmovima (npr. nazivi "
            "modula, predmeta, programa).\n"
            "Zadrzi smisao najnovijeg pitanja: ono i posle prepisa mora da "
            "pita isto sto i pre. Samo zameni nejasne reci konkretnim "
            "pojmovima i, kod elipticnog pitanja, dodaj pojam iz prethodnog "
            "pitanja — ne prepisuj prethodno pitanje i ne ubacuj njegove "
            "druge reci. "
            "Primer 1: prethodno pitanje 'Koje module ima smer Informatika?', "
            "odgovor navodi module, najnovije pitanje 'Sta je sa njima?' → "
            "'Sta je sa modulima Racunarske nauke i Softversko inzenjerstvo?'. "
            "Primer 2: prethodno pitanje 'Koliko traje master studija?', "
            "najnovije pitanje 'A koliko kosta?' → 'Koliko kosta master "
            "studija?'\n"
            "Ako je najnovije pitanje vec razumljivo samo po sebi, vrati ga "
            "nepromenjeno.\n\n"
            "Ne odgovaraj na pitanje. Vrati SAMO prepisano pitanje, nista drugo. "
            "Ne dodaj navodnike, objasnjenja ni komentare. Prepisi ga na srpskom "
            "latinskom pismu."
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
        )
        result = resp["response"].strip()
        self.last_rewrite["raw_response"] = result
        # Guard: if the model returned something nonsensical, fall back
        if not result or len(result) < 3:
            self.last_rewrite["returned"] = latest_question
            return latest_question
        self.last_rewrite["returned"] = result
        return result