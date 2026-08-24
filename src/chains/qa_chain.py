"""
qa_chain.py
~~~~~~~~~~~
University QA system — retrieves relevant chunks, formats citations,
and generates a grounded answer via the local LLM.

Multi-turn conversation support (v2)
------------------------------------
- Maintains per-session conversation history.
- Rewrites follow-up questions into standalone queries before embedding.
- Sends full message history to Ollama /api/chat with explicit num_ctx.
- Trims oldest turns when the token budget would be exceeded.
- Falls back to alternative retrieval strategies if the rewritten query
  returns no results.

Source references
-----------------
Every chunk in ChromaDB now carries a unified metadata schema:

    source   — full path or URL
    filename — human-readable name (basename or page title)
    type     — "pdf" | "docx" | "web"
    page     — 1-based page number for PDFs, 0 otherwise
    url      — canonical URL for web sources, "" for local files

_format_sources() turns that into human-readable citation lines:

    [1] lecture_notes.pdf — strana 4
    [2] Uvod u baze podataka.docx
    [3] Osnovi programiranja — https://example.com/article
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)

# ── rough token estimator (char-based, ~4 chars / token) ────────────── #
def _estimate_tokens(text: str) -> int:
    """Fast, zero-dependency token count approximation."""
    return max(1, len(text) // 4)


# ── referential / unclear question markers (Serbian Latin + English) ── #
# A question matching any of these cannot stand on its own — it refers to
# something mentioned earlier (usually the previous assistant answer).
_UNCLEAR_PATTERNS = [
    r"\bnjih\b", r"\bnjima\b", r"\bnjihov\w*", r"\bnjega\b", r"\bnjemu\b",
    r"\bih\b", r"\bga\b", r"\bmu\b",
    r"\bon\b", r"\bona\b", r"\bono\b", r"\boni\b", r"\bone\b",
    r"\bto\b", r"\bovo\b", r"\btim\b", r"\bte\b", r"\bti\b",
    r"\bthem\b", r"\bit\b", r"\bthis\b", r"\bthese\b", r"\bthose\b",
    r"\bthat\b",
    r"više informacij\w*", r"jos informacij\w*", r"više o\b",
    r"detaljnij\w*", r"opširnij\w*", r"objasni\w*", r"pojasni\w*",
    r"nabroj\w*", r"koji su to\b",
    r"more information", r"tell me more", r"explain\w*", r"elaborate\w*",
]

_UNCLEAR_REGEX = re.compile(
    r"(?i)" + "|".join(f"(?:{p})" for p in _UNCLEAR_PATTERNS)
)


class UniversityQASystem:
    # System prompt sent as the first message in every /api/chat call
    SYSTEM_PROMPT = (
        "Vi ste korisni univerzitetski asistent. Odgovarajte na pitanja "
        "na osnovu prilozenog konteksta iskljucivo na srpskom latinicnom pismu. "
        "Ako informacija nije u kontekstu, jasno to navedite. "
        "Kada koristite informacije iz konteksta, referencite izvore u obliku "
        "[Izvor N] unutar odgovora."
    )

    def __init__(self, llm_manager, vector_store, translator):
        self.llm_manager  = llm_manager
        self.vector_store = vector_store
        self.translator   = translator

        # Context-window ceiling (tokens) — derived from LLM config
        self.num_ctx = getattr(
            llm_manager, "num_ctx", getattr(llm_manager.config, "LLM_NUM_CTX", 8192)
        )

    # ------------------------------------------------------------------ #
    #  Source formatting                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_source(meta: dict, index: int) -> str:
        """
        Turn a single chunk's metadata dict into a numbered citation line.

        Examples
        --------
        [1] lecture_notes.pdf — strana 4
        [2] Uvod u baze podataka.docx
        [3] Osnovi programiranja — https://example.com/article
        """
        filename = meta.get("filename") or meta.get("source", "Nepoznat izvor")
        doc_type = meta.get("type", "")
        page     = meta.get("page", 0)
        url      = meta.get("url", "")

        parts = [f"[{index}] {filename}"]

        if doc_type == "pdf" and page:
            parts.append(f"strana {page}")

        if url:
            parts.append(url)

        return " — ".join(parts)

    def _format_sources(self, metadatas: list[dict]) -> list[str]:
        """
        Deduplicate and format all sources for a query result.
        Deduplication key: (source_path_or_url, page) so we don't list
        the same file page twice when two chunks came from the same page.
        """
        seen:    set[tuple] = set()
        sources: list[str]  = []

        for meta in metadatas:
            key = (meta.get("source", ""), meta.get("page", 0))
            if key in seen:
                continue
            seen.add(key)
            sources.append(self._format_source(meta, len(sources) + 1))

        return sources

    # ------------------------------------------------------------------ #
    #  Context building                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_context(documents: list[str], metadatas: list[dict]) -> str:
        """
        Interleave each chunk with a small inline source tag so the LLM
        can reference specific sources in its answer.

        Example:
            [Izvor 1 — lecture_notes.pdf, strana 4]
            <chunk text here>
        """
        parts: list[str] = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
            filename = meta.get("filename") or meta.get("source", "")
            page     = meta.get("page", 0)
            url      = meta.get("url", "")

            if page:
                label = f"{filename}, strana {page}"
            elif url:
                label = f"{filename} ({url})"
            else:
                label = filename

            parts.append(f"[Izvor {i} — {label}]\n{doc}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    #  Conversation history helpers                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_history_for_rewrite(
        history: list[dict[str, str]],
        include_last_answer: bool = False,
    ) -> str:
        """
        Convert the last 1–2 turns into a compact text block for the
        query-rewrite prompt.

        Assistant messages are truncated to 400 chars to keep the prompt
        focused on the topic rather than the full answer text.

        When ``include_last_answer`` is True (the latest question is
        unclear and refers back to the previous answer), the most recent
        assistant message is kept much longer (1200 chars) because it is
        the reference point that must resolve the question.

        Example output:
            User: Koliko ESPB ima predmet Uvod u programiranje?
            Assistant: Predmet Uvod u programiranje nosi 6 ESPB bodova...
        """
        if not history:
            return ""
        # Use at most the last 4 messages (2 full turns)
        recent = history[-4:]
        lines: list[str] = []
        for i, msg in enumerate(recent):
            role = msg.get("role", "user")
            label = "Korisnik" if role == "user" else "Asistent"
            content = msg.get("content", "")
            is_last = i == len(recent) - 1
            if role == "assistant":
                limit = 1200 if (include_last_answer and is_last) else 400
                if len(content) > limit:
                    content = content[:limit] + "..."
            lines.append(f"{label}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _is_unclear_question(question: str) -> bool:
        """
        True if the question cannot stand on its own — it contains
        pronouns or follow-up phrases ("more information", "explain",
        "njih", "them"...) that refer to earlier context.
        """
        return bool(_UNCLEAR_REGEX.search(question))

    # ------------------------------------------------------------------ #
    #  Query contextualization  (Fix #2)                                   #
    # ------------------------------------------------------------------ #

    def _standalone_question(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> str:
        """
        Rewrite the latest user question into a standalone, self-contained
        question using conversation history, so that embedding-based
        retrieval works correctly for follow-up questions.

        When the question is unclear (pronouns / "more information"-type
        follow-ups), the previous *assistant answer* is passed as the
        reference context so the rewrite resolves "them" → the actual
        entities from the previous answer (e.g. module names).

        If there is no history or the rewrite fails, the original question
        is returned unchanged.
        """
        if not history:
            return question

        unclear = self._is_unclear_question(question)
        history_text = self._format_history_for_rewrite(
            history, include_last_answer=unclear
        )
        if not history_text.strip():
            return question

        rewritten = self.llm_manager.rewrite_query(
            history_text, question, is_unclear=unclear
        )
        logger.info("Query rewrite (unclear=%s): %r → %r", unclear, question, rewritten)

        rewritten_stripped = rewritten.strip()

        if not rewritten_stripped:
            logger.warning("Rewrite empty, falling back to original question")
            return question

        if len(rewritten_stripped) < len(question) * 0.3:
            logger.warning("Rewrite too short, falling back to original question")
            return question

        if rewritten_stripped == question.strip():
            logger.warning("Rewrite unchanged, falling back to original question")
            return question

        if len(rewritten_stripped) > len(question) * 5:
            logger.warning("Rewrite suspiciously long (%d → %d chars), using original",
                           len(question), len(rewritten_stripped))
            return question

        return rewritten_stripped

    # ------------------------------------------------------------------ #
    #  Token-budget guard  (Fix #3)                                        #
    # ------------------------------------------------------------------ #

    def _trim_history(
        self,
        history: list[dict[str, str]],
        context: str,
        question: str,
    ) -> list[dict[str, str]]:
        """
        Ensure system_prompt + history + context + question fits inside
        the configured num_ctx by keeping the newest turns that fit.

        Messages that individually exceed the remaining budget are skipped
        rather than causing all older messages to be dropped.
        """
        system_tokens = _estimate_tokens(self.SYSTEM_PROMPT)
        # Account for the wrapping text added by _build_user_content
        user_wrapper = f"Kontekst:\n{context}\n\nPitanje: {question}"
        user_tokens = _estimate_tokens(user_wrapper)
        budget = int(self.num_ctx * 0.8) - system_tokens - user_tokens

        trimmed: list[dict[str, str]] = []
        for msg in reversed(history):
            msg_tokens = _estimate_tokens(msg.get("content", ""))
            if msg_tokens <= budget:
                trimmed.insert(0, msg)
                budget -= msg_tokens
        return trimmed

    # ------------------------------------------------------------------ #
    #  Message list builder                                                #
    # ------------------------------------------------------------------ #

    def _build_user_content(self, context: str, question: str) -> str:
        """Wrap the current question with retrieved context for the LLM."""
        return (
            f"Kontekst:\n{context}\n\n"
            f"Pitanje: {question}"
        )

    # ------------------------------------------------------------------ #
    #  Retrieval with fallback                                              #
    # ------------------------------------------------------------------ #

    def _retrieve_with_fallback(
        self,
        question: str,
        history: list[dict[str, str]],
        k: int = 5,
    ) -> tuple[list[str], list[dict]]:
        """
        Try multiple search strategies and return the first that yields
        results.  Returns (documents, metadatas) — both may be empty.
        """
        # ── Strategy 1: rewritten standalone question ─────────────────── #
        search_query = self._standalone_question(question, history)
        normalized   = self.translator.normalize_query(search_query)
        results      = self.vector_store.similarity_search(normalized, k=k)
        docs, metas  = results["documents"][0], results["metadatas"][0]
        if docs:
            logger.info("Retrieval OK (rewritten): %d docs", len(docs))
            return docs, metas

        # ── Strategy 2: original question verbatim ────────────────────── #
        logger.info("Fallback 1: trying original question verbatim")
        normalized = self.translator.normalize_query(question)
        results    = self.vector_store.similarity_search(normalized, k=k)
        docs, metas = results["documents"][0], results["metadatas"][0]
        if docs:
            logger.info("Retrieval OK (original): %d docs", len(docs))
            return docs, metas

        # ── Strategy 3: last assistant answer + current question ─────── #
        # For unclear follow-ups ("more info on them") the previous answer
        # contains the entities the question refers to.
        if history:
            last_answer = next(
                (h["content"] for h in reversed(history) if h.get("role") == "assistant"),
                "",
            )
            if last_answer and len(last_answer) > 10:
                combined = f"{last_answer[:800]} {question}"
                logger.info("Fallback 2: query + last answer %r", combined[:120])
                normalized = self.translator.normalize_query(combined)
                results    = self.vector_store.similarity_search(normalized, k=k)
                docs, metas = results["documents"][0], results["metadatas"][0]
                if docs:
                    logger.info("Retrieval OK (with last answer): %d docs", len(docs))
                    return docs, metas

        # ── Strategy 4: last user message + current question ──────────── #
        if history:
            last_user = next(
                (h["content"] for h in reversed(history) if h.get("role") == "user"),
                "",
            )
            if last_user and last_user != question:
                combined = f"{last_user} {question}"
                logger.info("Fallback 3: combined query %r", combined[:120])
                normalized = self.translator.normalize_query(combined)
                results    = self.vector_store.similarity_search(normalized, k=k)
                docs, metas = results["documents"][0], results["metadatas"][0]
                if docs:
                    logger.info("Retrieval OK (combined): %d docs", len(docs))
                    return docs, metas

        logger.warning("All retrieval strategies returned 0 documents")
        return [], []

    # ------------------------------------------------------------------ #
    #  Query  (non-streaming)                                              #
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        k: int = 5,
    ) -> dict:
        """
        Retrieve relevant chunks, build a grounded prompt, call the LLM,
        and return the answer alongside formatted source references.

        Parameters
        ----------
        question : str
            The latest user question.
        conversation_history : list[dict] | None
            Previous turns as [{"role": "user"|"assistant", "content": "..."}, ...].
        k : int
            Number of chunks to retrieve.

        Returns
        -------
        {
            "question": str,
            "answer":   str,
            "sources":  list[str],   # human-readable citation lines
        }
        """
        history = conversation_history or []

        # ── Step 1: Retrieve with fallback ───────────────────────────── #
        documents, metadatas = self._retrieve_with_fallback(question, history, k=k)

        if not documents:
            return {
                "question": question,
                "answer":   "Nisam pronasao relevantne informacije u bazi znanja za ovo pitanje.",
                "sources":  [],
            }

        # ── Step 2: Build messages with full history ──────────────────── #
        context  = self._build_context(documents, metadatas)
        history  = self._trim_history(history, context, question)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]
        messages.extend(history)
        messages.append({
            "role": "user",
            "content": self._build_user_content(context, question),
        })

        # ── Step 3: Call LLM via /api/chat ────────────────────────────── #
        response = self.llm_manager.chat(messages)
        sources  = self._format_sources(metadatas)

        return {
            "question": question,
            "answer":   response,
            "sources":  sources,
        }

    # ------------------------------------------------------------------ #
    #  Query  (streaming)                                                  #
    # ------------------------------------------------------------------ #

    async def query_stream(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream response tokens — sources are yielded as a trailing block.

        Parameters
        ----------
        question : str
            The latest user question.
        conversation_history : list[dict] | None
            Previous turns as [{"role": "user"|"assistant", "content": "..."}, ...].
        """
        history = conversation_history or []

        # ── Step 1: Retrieve with fallback ───────────────────────────── #
        documents, metadatas = self._retrieve_with_fallback(question, history)

        if not documents:
            yield "Nisam pronasao relevantne informacije u bazi znanja za ovo pitanje."
            return

        # ── Step 2: Build messages with full history ──────────────────── #
        context = self._build_context(documents, metadatas)
        history = self._trim_history(history, context, question)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]
        messages.extend(history)
        messages.append({
            "role": "user",
            "content": self._build_user_content(context, question),
        })

        # ── Step 3: Stream from /api/chat ─────────────────────────────── #
        for token in self.llm_manager.chat_stream(messages):
            yield token

        # ── Step 4: Append sources block ──────────────────────────────── #
        sources = self._format_sources(metadatas)
        if sources:
            yield "\n\n---\nIzvori:\n" + "\n".join(sources)