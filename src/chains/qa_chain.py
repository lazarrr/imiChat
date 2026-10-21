"""
qa_chain.py
~~~~~~~~~~~
University QA system — retrieves relevant chunks, formats citations,
and generates a grounded answer via the local LLM.

Multi-turn conversation support (v2)
------------------------------------
- The answering LLM sees only the system prompt, the retrieved chunks and
  one question — never the conversation history.
- Per-session history is kept only to rewrite *unclear* follow-ups:
  questions that point back at the previous answer ("njih", "o njima",
  "njega", "više o tome", "A koliko…") are rewritten into a standalone
  question, which is used both for retrieval and as the question the LLM
  answers.  The rewrite is the dedicated RAG rewrite prompt
  (``QUERY_REWRITE_PROMPT`` in ``llm_manager.py``): it receives the last
  1–2 turns, must never answer, resolves pronouns and short follow-ups
  from that history, keeps an already standalone question unchanged and
  returns only the question.  Clear questions are searched and answered
  exactly as typed.
- Unclear follow-ups fall back to history-assisted retrieval strategies
  (previous question, then previous answer) if the rewritten query
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

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import AsyncGenerator, Iterator

from src.processing.serbian_text import SerbianAnswerNormalizer
from src.processing.translator import normalize_script

logger = logging.getLogger(__name__)

# ── referential / unclear question markers (Serbian Latin + English) ── #
# A question matching any of these cannot stand on its own — it refers to
# something mentioned earlier (usually the previous assistant answer).
# Only such questions are rewritten with the conversation history.  The
# check runs on the Latin-normalized question, so Cyrillic input matches
# too; [šs]-style classes cover questions typed without diacritics.
_UNCLEAR_PATTERNS = [
    r"\bnjih\b", r"\bnjima?\b", r"\bnjihov\w*", r"\bnjeg\w*", r"\bnjemu\b",
    r"\bnje\b", r"\bnju\b", r"\bnjoj\b", r"\bnjom\b", r"\bnjen\w*",
    r"\bih\b", r"\bim\b", r"\bga\b", r"\bmu\b", r"\bjoj\b",
    r"\bon\b", r"\bona\b", r"\bono\b", r"\boni\b", r"\bone\b",
    r"\bto\b", r"\bovo\b", r"\btim\b", r"\bte\b", r"\bti\b",
    r"\btaj\b", r"\btoga?\b", r"\btom[eu]?\b", r"\btih\b", r"\btima\b",
    r"\bnavedeni\w*", r"\bpomenut\w*", r"\bprethodn\w*\s+odgovor\w*",
    r"\bthem\b", r"\bthey\b", r"\btheir\b",
    r"\bit\b", r"\bthis\b", r"\bthese\b", r"\bthose\b", r"\bthat\b",
    r"vi[šs]e informacij\w*", r"jo[šs] informacij\w*",
    r"vi[šs]e o\b", r"jo[šs] o\b",
    # "može malo više?" — a request for more detail, named in the rewrite prompt
    r"\bmalo vi[šs]e\b",
    r"detaljnij\w*", r"op[šs]irnij\w*", r"objasni\w*", r"pojasni\w*",
    r"nabroj\w*", r"koji su to\b",
    r"[šs]ta je sa\b", r"kako je sa\b", r"[šs]ta jo[šs]\b",
    # elliptical follow-ups: "A koliko traje?", "I šta još?", "Pa kada?"
    r"^\s*(?:a|i|pa)\s",
    r"more information", r"tell me more", r"explain\w*", r"elaborate\w*",
    r"what about\b",
]

_UNCLEAR_REGEX = re.compile(
    r"(?i)" + "|".join(f"(?:{p})" for p in _UNCLEAR_PATTERNS)
)

# ── "no information" answers ─────────────────────────────────────────── #
# When the context does not hold the answer, the system prompt makes the
# model say so ("Nemam informaciju o tome.") and show no sources, so such
# an answer gets no Izvori block.  A long answer, or one that still cites
# a source ("[Izvor 2]"), answers at least part of the question and keeps
# its sources.
_NO_INFO_REGEX = re.compile(
    r"(?i)\bnema(?:m|mo)?\b[^.!?\n]{0,40}\b(?:informacij|poda[tc])"
    r"|\bne (?:posedujem|postoj\w*)\b[^.!?\n]{0,40}\b(?:informacij|poda[tc])"
    r"|\bnisam\b[^.!?\n]{0,30}\b(?:prona|na[šs]a)\w*"
    r"|\bne mogu (?:da )?(?:odgovor|prona)\w*"
    r"|\bkontekst\w*[^.!?\n]{0,40}\bne (?:sadr[žz]i|navodi|pominje)"
    r"|\bnije (?:navedeno|pomenuto|spomenuto)\b"
    r"|\b(?:don'?t|do not) have (?:any )?(?:information|data)\b"
    r"|\bno information\b"
)
_CITATION_REGEX = re.compile(r"(?i)\bizvor\w*\s*\d|\[\d+\]")
_NO_INFO_MAX_CHARS = 300

# What next() returns once _stream_answer() is exhausted (see query_stream).
_END_OF_STREAM = object()


def _source_key(meta: dict) -> tuple:
    """Identity of a cited source: one PDF page or one web-page section."""
    return (meta.get("source", ""), meta.get("page", 0))


class UniversityQASystem:
    # System prompt sent as the first message in every /api/chat call.
    #
    # The prompt itself is written in correct Serbian Latin with diacritics:
    # a prompt full of "latinicno", "prekidas" and "referneciraj" teaches a
    # small model exactly the wrong spelling.  The explicit script rule and
    # the output normaliser guarantee the script even if the model ignores
    # it.  Keep the structure simple: a heavily restructured prompt made
    # this quantised Qwen model degenerate into mixed-script output.
    SYSTEM_PROMPT = (
        "Vi ste korisni univerzitetski asistent. Odgovarajte na pitanja "
        "isključivo na srpskom jeziku i latiničnim pismom (č, ć, š, ž, đ).\n\n"
        "PISMO I JEZIK:\n"
        "- Uvek pišite latinicom. Nikada ne prelazite na ćirilicu, čak i kada "
        "su pitanje ili kontekst na ćirilici — ćirilične delove prenesite "
        "latinicom.\n"
        "- Poštujte srpski pravopis: ne izostavljajte dijakritike (č, ć, š, ž, "
        "đ) i ne izmišljajte reči. Pazite na rod, broj i padež.\n"
        "- Koristite srpski ekavski izgovor i srpske oblike reči, ne hrvatske "
        "ili ijekavske: smer (ne smjer), vreme (ne vrijeme), mesto (ne "
        "mjesto), četvorogodišnji (ne četverogodišnji), studijskog (ne "
        "studijskoga).\n"
        "- Ne ubacujte engleske reči i izraze; napišite „kao i“, ne „kao well "
        "kao“.\n"
        "- Koristite nazive tačno onako kako su napisani u kontekstu; ne "
        "menjajte ih i ne prevodite ih.\n"
        "- Skraćenice: OAS = osnovne akademske studije, ESPB = evropski sistem "
        "prenosa bodova. Pišite „240 ESPB“ ili „240 ESPB bodova“, a ne "
        "„240 evropskih sistema prenosa bodova“.\n"
        "- Česte greške ispravljajte ovako: studijama (ne studiama), prijemni "
        "ispit (ne prijemi ispit), prijemnih ispita (ne prijmnih ispita), "
        "Softversko inženjerstvo (ne inženjering), informacioni servis / "
        "informacionog servisa (ne informacijski), najkasnije (ne najkasnje), "
        "osnovnih (ne osnivnih), kratak vodič (ne kratik vodić), studentima / "
        "studenata (ne studenta).\n"
        "- Odgovarajte direktno. Ne pominjite kontekst ni dokumentaciju; "
        "odmah navedite odgovor.\n\n"
        "AKO INFORMACIJA NIJE DOSTUPNA:\n"
        "- Ako informaciju ne možete da pronađete, napišite samo „Nemam "
        "informaciju o tome.“, bez objašnjenja i bez pominjanja konteksta. "
        "Nemojte nagađati i nemojte navoditi izvore.\n\n"
        "IZVORI:\n"
        "- Kada koristite informacije iz priloženih izvora, navedite izvor u "
        "obliku [Izvor N], a ne [N].\n"
        "- Zaglavlje svakog izvora navodi verziju studijskog programa i njen "
        "status (npr. „akreditacija 2026 | aktuelni program (upis u toku)“ ili "
        "„akreditacija 2019 | prethodni program“). Ako kontekst sadrži podatke "
        "za više verzija studijskog programa, za svaki podatak navedite na "
        "koju se verziju odnosi i prvo navedite program za koji je upis u "
        "toku.\n\n"
        "PRIMER:\n"
        "Pitanje: „Koliko traje OAS Informatika?“\n"
        "Odgovor: „Studije traju četiri godine [Izvor 1].“"
    )

    def __init__(self, llm_manager, vector_store, translator):
        self.llm_manager  = llm_manager
        self.vector_store = vector_store
        self.translator   = translator

    # ------------------------------------------------------------------ #
    #  Script helpers                                                      #
    # ------------------------------------------------------------------ #

    def _to_latinic(self, text: str) -> str:
        """Convert text to Serbian Latin with the configured translator."""
        if not text:
            return text
        for name in ("to_latinic", "cyrillic_to_latinic", "normalize_query"):
            converter = getattr(self.translator, name, None)
            if callable(converter):
                converted = converter(text)
                return converted if isinstance(converted, str) else text
        return text

    def _to_script(self, text: str, script: str = "latinic") -> str:
        """Return text in the requested output script (Latin by default)."""
        if not text:
            return text
        if normalize_script(script) == "cyrillic":
            converter = getattr(self.translator, "latinic_to_cyrillic", None)
            if callable(converter):
                return converter(self._to_latinic(text))
        return self._to_latinic(text)

    def _filename_in_script(self, filename: str, script: str) -> str:
        """
        Transliterate a file name for display, keeping its extension.

        A full Latin→Cyrillic conversion would turn ``.pdf`` into ``.пдф``;
        the suffix is machine-facing, so it stays ASCII.
        """
        latin = self._to_latinic(filename)
        if normalize_script(script) != "cyrillic":
            return latin
        stem, dot, suffix = latin.rpartition(".")
        if dot and suffix and len(suffix) <= 5 and suffix.isalnum():
            return f"{self._to_script(stem, script)}.{suffix}"
        return self._to_script(latin, script)

    def _answer_normalizer(
        self, documents: list[str], script: str = "latinic",
    ) -> SerbianAnswerNormalizer:
        """Build the per-answer normaliser, using the chunks as vocabulary."""
        return SerbianAnswerNormalizer(self.translator, documents, script)

    # ------------------------------------------------------------------ #
    #  Source formatting                                                   #
    # ------------------------------------------------------------------ #

    def _format_source(
        self, meta: dict, index: int, script: str = "latinic",
    ) -> str:
        """
        Turn a single chunk's metadata dict into a numbered citation line.

        The file name follows the requested output script; an URL is kept
        verbatim so it stays clickable.

        Examples
        --------
        [1] lecture_notes.pdf — strana 4
        [2] Uvod u baze podataka.docx
        [3] Osnovi programiranja — https://example.com/article
        """
        filename = self._filename_in_script(
            meta.get("filename") or meta.get("source", "Nepoznat izvor"),
            script,
        )
        doc_type = meta.get("type", "")
        page     = meta.get("page", 0)
        url      = meta.get("url", "")

        parts = [f"[{index}] {filename}"]

        if doc_type == "pdf" and page:
            page_label = "страна" if normalize_script(script) == "cyrillic" else "strana"
            parts.append(f"{page_label} {page}")

        if url:
            parts.append(url)

        return " — ".join(parts)

    def _format_sources(
        self, metadatas: list[dict], script: str = "latinic",
    ) -> list[str]:
        """
        Deduplicate and format all sources for a query result.
        Deduplication key: _source_key() (source path / URL + page), so we
        don't list the same file page twice when two chunks came from the
        same page.  Numbers follow first appearance, exactly like the
        [Izvor N] tags _build_context() puts in front of the chunks.
        """
        seen:    set[tuple] = set()
        sources: list[str]  = []

        for meta in metadatas:
            key = _source_key(meta)
            if key in seen:
                continue
            seen.add(key)
            sources.append(self._format_source(meta, len(sources) + 1, script))

        return sources

    @staticmethod
    def _is_no_info_answer(answer: str) -> bool:
        """
        True if the answer only says that the context has no information
        on the question (see _NO_INFO_REGEX) — it must not list sources.
        """
        answer = answer.strip()
        return (
            len(answer) <= _NO_INFO_MAX_CHARS
            and bool(_NO_INFO_REGEX.search(answer))
            and not _CITATION_REGEX.search(answer)
        )

    def _answer_sources(
        self, answer: str, metadatas: list[dict], script: str = "latinic",
    ) -> list[str]:
        """
        Citation lines for an answer — none for a "no information" answer.

        The no-information check runs on the Latin form, so it works even
        when the client requested Cyrillic output.
        """
        if self._is_no_info_answer(self._to_latinic(answer)):
            logger.info("No-information answer — sources omitted")
            return []
        return self._format_sources(metadatas, script)

    # ------------------------------------------------------------------ #
    #  Context building                                                    #
    # ------------------------------------------------------------------ #

    def _build_context(self, documents: list[str], metadatas: list[dict]) -> str:
        """
        Interleave each chunk with a small inline source tag so the LLM
        can reference specific sources in its answer.

        Chunks from the same source share one number, in order of first
        appearance — the numbering _format_sources() uses, so "[Izvor 2]"
        in an answer is line [2] of the Izvori block.

        The answering model always receives Latin text, even when a chunk
        was indexed in Cyrillic: a Cyrillic context is the strongest trigger
        for the model to switch scripts.

        Example:
            [Izvor 1 — lecture_notes.pdf, strana 4]
            <chunk text here>
        """
        numbers: dict[tuple, int] = {}
        parts:   list[str]        = []
        for doc, meta in zip(documents, metadatas):
            number   = numbers.setdefault(_source_key(meta), len(numbers) + 1)
            filename = self._to_latinic(
                meta.get("filename") or meta.get("source", "")
            )
            page     = meta.get("page", 0)
            url      = meta.get("url", "")

            if page:
                label = f"{filename}, strana {page}"
            elif url:
                label = f"{filename} ({url})"
            else:
                label = filename

            parts.append(f"[Izvor {number} — {label}]\n{self._to_latinic(doc)}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    #  Optional pipeline trace                                             #
    # ------------------------------------------------------------------ #
    #
    # Every entry point takes an optional ``trace`` dict.  When the caller
    # passes one, the chain fills it in as it goes: which question was
    # searched for and why, which queries were tried, which chunks came back
    # and what prompt the answering model saw.  The API turns that into one
    # JSON line per turn (src/api/request_log.py).  The chain itself never
    # writes files, so the CLI and the tests are unaffected.

    @staticmethod
    def _record_chunks(trace: dict | None, documents: list[str],
                       metadatas: list[dict], distances: list | None = None,
                       excluded: list[dict] | None = None,
                       excluded_total: int | None = None) -> None:
        """
        Put the retrieved chunks into ``trace``, with the metadata and the
        similarity distance that made each one relevant.

        Distances come from ChromaDB (lower = closer); they are absent for
        keyword-only hits and when the store does not report them.

        ``metadatas`` is stored whole under ``metadata`` as well as being
        flattened into the common keys: the log must show every field the
        vector store holds (version labels, scraper flags, chunk hashes…),
        including ones this project does not know about yet.

        ``excluded`` holds chunks that were retrieved but dropped before the
        prompt (threshold, legacy version, duplicate, reranker…); the
        visualizer shows them separately so it is clear what the model did
        *not* get.
        """
        if trace is None:
            return
        distances = distances or [None] * len(documents)
        trace["chunks"] = [
            {
                "rank": i + 1,
                "source": meta.get("source", ""),
                "filename": meta.get("filename", ""),
                "type": meta.get("type", ""),
                "page": meta.get("page", 0),
                "url": meta.get("url", ""),
                "distance": dist,
                "chars": len(doc),
                # Everything the vector store attached to the chunk.
                "metadata": dict(meta),
                "text": doc,
            }
            for i, (doc, meta, dist) in enumerate(
                zip(documents, metadatas, distances)
            )
        ]
        if excluded is not None:
            trace["chunks_excluded"] = list(excluded)
        if excluded_total is not None:
            trace["chunks_excluded_total"] = excluded_total

    @staticmethod
    def _record_messages(trace: dict | None, messages: list[dict[str, str]]) -> None:
        """The exact message list sent to the answering model (no history)."""
        if trace is None:
            return
        trace["prompt"] = {
            "messages": messages,
            "system_prompt": next(
                (m["content"] for m in messages if m["role"] == "system"), ""
            ),
            "char_count": sum(len(m["content"]) for m in messages),
        }

    @contextlib.contextmanager
    def _stage(self, trace: dict | None, name: str) -> Iterator[None]:
        """
        Time one pipeline stage into ``trace['timings_ms'][name]``.

        Stage times are what make a slow turn diagnosable from the log file:
        the rewrite is a separate LLM call, so a turn can spend seconds
        there before retrieval even starts.
        """
        if trace is None:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            timings = trace.setdefault("timings_ms", {})
            timings[name] = round((time.perf_counter() - started) * 1000, 1)

    # ------------------------------------------------------------------ #
    #  Conversation history helpers                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_history_for_rewrite(history: list[dict[str, str]]) -> str:
        """
        Convert the last 1–2 turns into a compact text block for the
        query-rewrite prompt.

        Both parts of the last turn are reference points for the rewrite:
        the previous *question* names the topic (which an elliptical
        follow-up like "A koliko traje?" leaves out) and the previous
        *answer* names the entities a pronoun refers to.  The most recent
        assistant message therefore keeps its length (1200 chars), every
        user question is kept up to 500 chars, and older assistant messages
        are cut to 400 chars to keep the prompt focused on the topic rather
        than the full answer text.

        Example output:
            Korisnik: Koliko ESPB ima predmet Uvod u programiranje?
            Asistent: Predmet Uvod u programiranje nosi 6 ESPB bodova...
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
            if role == "user":
                # Questions are short and dense — the reference to resolve.
                limit = 500
            elif i == len(recent) - 1:
                limit = 1200
            else:
                limit = 400
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
        self, question: str, history: list[dict[str, str]],
        trace: dict | None = None,
    ) -> str:
        """
        Rewrite an unclear follow-up question into a standalone,
        self-contained question using the conversation history.

        The rewrite may draw on the whole last turn: the previous
        *assistant answer* names the entities a pronoun refers to
        ("them" → module names), while the previous *user question* carries
        the topic an elliptical follow-up ("A koliko traje?") inherits.
        The result is used both for retrieval and as the question the
        answering LLM sees — the LLM never sees the history itself, so the
        rewrite must name both the entities and the topic.

        If the rewrite fails, the original question is returned unchanged.

        With ``trace``, the history text, the full rewrite prompt, the raw
        model output, the guard that was applied and the accepted result are
        all recorded there.
        """
        returned = question
        rejected: str | None = None
        reason = "not_attempted"
        raw: str | None = None
        rewritten_stripped = ""
        history_text = ""

        with self._stage(trace, "rewrite"):
            history_text = self._to_latinic(
                self._format_history_for_rewrite(history)
            )
            if not history_text.strip():
                reason = "no_history"
            else:
                rewritten = self.llm_manager.rewrite_query(history_text, question)
                # The rewrite inherits the model's script habits; force it
                # back to Latin before the guards compare it with the input.
                rewritten = self._to_latinic(rewritten)
                details = getattr(self.llm_manager, "last_rewrite", None) or {}
                raw = details.get("raw_response", rewritten)
                logger.info("Query rewrite: %r → %r", question, rewritten)
                rewritten_stripped = rewritten.strip()

                if not rewritten_stripped:
                    reason, rejected = "empty", rewritten_stripped
                    logger.warning("Rewrite empty, falling back to original question")
                elif len(rewritten_stripped) < len(question) * 0.3:
                    reason, rejected = "too_short", rewritten_stripped
                    logger.warning("Rewrite too short, falling back to original question")
                elif rewritten_stripped == question.strip():
                    reason = "unchanged"
                    logger.info("Rewrite unchanged — question is already standalone")
                # Naming the referenced entities legitimately makes a short
                # question ("A o njima?") many times longer — only reject
                # output long enough to be an answer rather than a question.
                elif len(rewritten_stripped) > max(len(question) * 5, 300):
                    reason, rejected = "too_long", rewritten_stripped
                    logger.warning(
                        "Rewrite suspiciously long (%d → %d chars), using original",
                        len(question), len(rewritten_stripped),
                    )
                else:
                    reason, returned = "accepted", rewritten_stripped

        if trace is not None:
            details = getattr(self.llm_manager, "last_rewrite", None) or {}
            trace["rewrite"] = {
                "question_in": question,
                "history_text": details.get("history_text", history_text),
                "prompt": details.get("prompt"),
                "raw_response": raw,
                "question_out": returned,
                "changed": returned != question,
                "result": reason,
                # The rejected candidate is worth keeping: a guard that
                # throws away good rewrites shows up here as a repeating
                # pattern, not as a single bad answer.
                "rejected": rejected,
            }

        return returned

    def _resolve_question(
        self, question: str, history: list[dict[str, str]],
        trace: dict | None = None,
    ) -> tuple[str, bool]:
        """
        Decide which question is searched for and sent to the LLM.

        Returns ``(question_to_use, unclear)``.  Only an unclear follow-up
        (see ``_UNCLEAR_PATTERNS``) with existing history is rewritten;
        every other question is used as typed, converted to Latin because
        that is what the answering model must receive.

        With ``trace``, the normalization and the unclear-verdict (with the
        pattern that matched) are recorded — that pair is what explains why
        a question was or was not rewritten.
        """
        normalized = self._to_latinic(question)
        match = _UNCLEAR_REGEX.search(normalized)
        has_history = bool(history)

        if not has_history or match is None:
            if trace is not None:
                trace["rewrite"] = {
                    "question_in": question,
                    "normalized": normalized,
                    "history_text": self._to_latinic(
                        self._format_history_for_rewrite(history)
                    ) if has_history else "",
                    "prompt": None,
                    "raw_response": None,
                    "question_out": normalized,
                    "changed": False,
                    "result": "skipped_no_history" if not has_history
                              else "skipped_clear_question",
                    "matched_pattern": match.group(0) if match else None,
                    "rejected": None,
                }
            return normalized, False

        search_question = self._standalone_question(normalized, history, trace)
        if trace is not None:
            trace["rewrite"]["normalized"] = normalized
            trace["rewrite"]["matched_pattern"] = match.group(0)
        return self._to_latinic(search_question), True

    # ------------------------------------------------------------------ #
    #  Message list builder                                                #
    # ------------------------------------------------------------------ #

    def _build_user_content(self, context: str, question: str) -> str:
        """Wrap the current question with retrieved context for the LLM."""
        return (
            f"Kontekst:\n{context}\n\n"
            f"Pitanje: {question}"
        )

    def _build_messages(self, context: str, question: str) -> list[dict[str, str]]:
        """
        Messages for the answering LLM: the system prompt and one user
        message with the retrieved context and the question.  The
        conversation history is deliberately left out — it only feeds the
        query rewrite.
        """
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_content(context, question)},
        ]

    # ------------------------------------------------------------------ #
    #  Retrieval with fallback                                              #
    # ------------------------------------------------------------------ #

    def _search(
        self, query: str, k: int | None, trace: dict | None = None,
        strategy: str = "", rank: int = 0,
    ) -> tuple[list[str], list[dict], list]:
        """
        Normalize the query (Cyrillic → Latin) and run a similarity search.
        ``k=None`` lets the vector store use TOP_K_RESULTS (clamped to
        MAX_CONTEXT_CHUNKS).

        Returns ``(documents, metadatas, distances)``.  With ``trace``, the
        attempt is appended to ``trace['retrieval']['attempts']`` — including
        the attempts that found nothing, which is what shows whether a
        rewrite sent retrieval in the wrong direction.

        With ``trace`` the store is also asked for diagnostics, so chunks it
        dropped on the way (distance threshold, legacy version, duplicate,
        reranker) are recorded under ``trace['retrieval']['excluded']``
        instead of disappearing without trace.
        """
        diagnostics = trace is not None
        normalized = self.translator.normalize_query(query)
        results = self.vector_store.similarity_search(
            normalized, k=k, diagnostics=diagnostics
        )
        docs       = results["documents"][0]
        metas      = results["metadatas"][0]
        distances  = (results.get("distances") or [[]])[0]

        if trace is not None:
            retrieval = trace.setdefault("retrieval", {})
            retrieval.setdefault("attempts", []).append({
                "rank": rank,
                "strategy": strategy,
                "query": normalized,
                "query_chars": len(normalized),
                "hits": len(docs),
                "top_distance": min(
                    (d for d in distances if d is not None), default=None
                ),
                "excluded": len(results.get("excluded") or []),
            })
            # Each excluded chunk remembers which search attempt produced it:
            # the winner's list is the one about the answered question, the
            # losers' lists belong to queries that were tried and rejected.
            for chunk in results.get("excluded") or []:
                retrieval.setdefault("excluded", []).append({
                    **chunk, "attempt_rank": rank, "attempt_strategy": strategy,
                })
            if results.get("excluded_total") is not None:
                retrieval["excluded_total"] = (
                    retrieval.get("excluded_total", 0) + results["excluded_total"]
                )
        return docs, metas, distances

    def _retrieve_with_fallback(
        self,
        search_question: str,
        question: str,
        history: list[dict[str, str]],
        unclear: bool,
        k: int | None = None,
        trace: dict | None = None,
    ) -> tuple[list[str], list[dict], list]:
        """
        Try multiple search strategies and return the first that yields
        results.  Returns (documents, metadatas, distances) — all may be
        empty.

        ``search_question`` is the question the LLM will answer (the
        rewrite for an unclear follow-up, otherwise the original).  Only
        unclear follow-ups fall back to history-assisted queries — the last
        user question first (it carries the topic), then the last answer
        (it names the referenced entities); a clear question never depends
        on the conversation history.

        With ``trace``, every attempt is recorded along with the name of the
        strategy that won (``trace['retrieval']['winning_strategy']``), or
        ``"none"`` when nothing was found.  Chunks the store dropped are kept
        too, marked with the attempt that produced them and whether that
        attempt was the one whose question was answered.
        """
        def finish(docs, metas, distances, strategy: str):
            if trace is not None:
                retrieval = trace["retrieval"]
                winning = strategy if docs else "none"
                retrieval["winning_strategy"] = winning
                # An excluded chunk from a losing attempt belongs to a query
                # that was tried and rejected; one from the winning attempt
                # belongs to the question that was actually answered.
                for chunk in retrieval.get("excluded", []):
                    chunk["attempt_kept"] = chunk.get("attempt_strategy") == winning
            return docs, metas, distances

        # ── Strategy 1: the question that will be answered ────────────── #
        docs, metas, distances = self._search(
            search_question, k, trace, "search_question", 1)
        if docs:
            logger.info("Retrieval OK: %d docs", len(docs))
            return finish(docs, metas, distances, "search_question")

        if not unclear:
            logger.warning("Retrieval returned 0 documents")
            return finish([], [], [], "none")

        # ── Strategy 2: original question verbatim ────────────────────── #
        attempt = 2
        if search_question != question:
            logger.info("Fallback 1: trying original question verbatim")
            docs, metas, distances = self._search(
                question, k, trace, "original_question", attempt)
            if docs:
                logger.info("Retrieval OK (original): %d docs", len(docs))
                return finish(docs, metas, distances, "original_question")
            attempt += 1

        # ── Strategy 3: last user question + current question ─────────── #
        # The previous question carries the topic the follow-up points back
        # at ("A koliko traje?" after a question about a course), so it is
        # tried before the previous answer, which is longer and noisier.
        last_user = next(
            (h["content"] for h in reversed(history) if h.get("role") == "user"),
            "",
        )
        if last_user and last_user != question:
            combined = f"{last_user} {question}"
            logger.info("Fallback 2: combined query %r", combined[:120])
            docs, metas, distances = self._search(
                combined, k, trace, "previous_question_plus_current", attempt)
            if docs:
                logger.info("Retrieval OK (combined): %d docs", len(docs))
                return finish(docs, metas, distances, "previous_question_plus_current")
            attempt += 1

        # ── Strategy 4: last assistant answer + current question ──────── #
        # When the previous question does not name what the follow-up
        # refers to, the answer usually does ("more info on them").
        last_answer = next(
            (h["content"] for h in reversed(history) if h.get("role") == "assistant"),
            "",
        )
        if last_answer and len(last_answer) > 10:
            combined = f"{last_answer[:800]} {question}"
            logger.info("Fallback 3: query + last answer %r", combined[:120])
            docs, metas, distances = self._search(
                combined, k, trace, "previous_answer_plus_current", attempt)
            if docs:
                logger.info("Retrieval OK (with last answer): %d docs", len(docs))
                return finish(docs, metas, distances, "previous_answer_plus_current")

        logger.warning("All retrieval strategies returned 0 documents")
        return finish([], [], [], "none")

    # ------------------------------------------------------------------ #
    #  Query  (non-streaming)                                              #
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        k: int | None = None,
        trace: dict | None = None,
        script: str = "latinic",
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
            Used only to rewrite unclear follow-up questions — it is never
            sent to the answering LLM.
        k : int | None
            Number of chunks to retrieve (default: TOP_K_RESULTS, never more
            than MAX_CONTEXT_CHUNKS).
        trace : dict | None
            Optional dict filled with the pipeline trace — the rewrite, every
            retrieval attempt, the chunks and the prompt sent to the model
            (see the "Optional pipeline trace" section).
        script : str
            Output script: ``"latinic"`` (default) or ``"cyrillic"``.  The
            model itself always writes Latin; the answer is transliterated
            deterministically at the end.

        Returns
        -------
        {
            "question": str,
            "answer":   str,
            "sources":  list[str],   # human-readable citation lines
        }
        """
        history = conversation_history or []
        script = normalize_script(script)
        if trace is not None:
            trace["script"] = script

        # ── Step 1: Rewrite the question if it refers back ────────────── #
        search_question, unclear = self._resolve_question(question, history, trace)

        # ── Step 2: Retrieve with fallback ───────────────────────────── #
        with self._stage(trace, "retrieval"):
            documents, metadatas, distances = self._retrieve_with_fallback(
                search_question, question, history, unclear, k=k, trace=trace
            )

        if trace is not None:
            retrieval = trace.get("retrieval") or {}
            trace["search_question"] = search_question
            trace["unclear"] = unclear
            self._record_chunks(
                trace, documents, metadatas, distances,
                excluded=retrieval.get("excluded"),
                excluded_total=retrieval.get("excluded_total"),
            )

        if not documents:
            fallback = (
                "Nisam pronašao relevantne informacije u bazi znanja "
                "za ovo pitanje."
            )
            return {
                "question": question,
                "answer":   self._to_script(fallback, script),
                "sources":  [],
            }

        # ── Step 3: Build messages — question + chunks, no history ────── #
        context  = self._build_context(documents, metadatas)
        messages = self._build_messages(context, search_question)
        logger.info("Answering %r with %d chunk(s)", search_question, len(documents))
        self._record_messages(trace, messages)

        # ── Step 4: Call LLM via /api/chat ────────────────────────────── #
        normalizer = self._answer_normalizer(documents, script)
        with self._stage(trace, "answer"):
            response = normalizer.normalize(self.llm_manager.chat(messages))
        sources  = self._answer_sources(response, metadatas, script)
        if trace is not None:
            trace["sources"] = sources

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
        trace: dict | None = None,
        script: str = "latinic",
    ) -> AsyncGenerator[str, None]:
        """
        Stream response tokens — sources are yielded as a trailing block.

        Every step is a blocking call (LLM HTTP for the rewrite and the
        answer, the E5 model and Chroma for retrieval), so each token of
        _stream_answer() is fetched in a worker thread.  Run on the event
        loop, those calls would hold up every other request — and /health —
        until the whole answer had streamed.

        Parameters
        ----------
        question : str
            The latest user question.
        conversation_history : list[dict] | None
            Previous turns as [{"role": "user"|"assistant", "content": "..."}, ...].
            Used only to rewrite unclear follow-up questions — it is never
            sent to the answering LLM.
        trace : dict | None
            Optional dict filled with the pipeline trace (see the "Optional
            pipeline trace" section).  Complete once the stream is
            exhausted, because ``timings_ms['total']`` is written last.
        script : str
            Output script: ``"latinic"`` (default) or ``"cyrillic"``.
        """
        # A copy: the session store may append to its list while a worker
        # thread is still reading this one.
        started = time.perf_counter()
        tokens = self._stream_answer(
            question, list(conversation_history or []), trace, script=script,
        )
        try:
            while True:
                token = await asyncio.to_thread(next, tokens, _END_OF_STREAM)
                if token is _END_OF_STREAM:
                    return
                yield token
        finally:
            # Runs on early client disconnect too, so a turn that was cut off
            # is still logged with the part of the pipeline that did run.
            if trace is not None:
                trace.setdefault("timings_ms", {})["total"] = round(
                    (time.perf_counter() - started) * 1000, 1
                )

    def _stream_answer(
        self, question: str, history: list[dict[str, str]],
        trace: dict | None = None, script: str = "latinic",
    ) -> Iterator[str]:
        """Blocking generator behind query_stream()."""
        script = normalize_script(script)
        if trace is not None:
            trace["script"] = script

        # ── Step 1: Rewrite the question if it refers back ────────────── #
        search_question, unclear = self._resolve_question(question, history, trace)

        # ── Step 2: Retrieve with fallback ───────────────────────────── #
        with self._stage(trace, "retrieval"):
            documents, metadatas, distances = self._retrieve_with_fallback(
                search_question, question, history, unclear, trace=trace
            )

        if trace is not None:
            retrieval = trace.get("retrieval") or {}
            trace["search_question"] = search_question
            trace["unclear"] = unclear
            self._record_chunks(
                trace, documents, metadatas, distances,
                excluded=retrieval.get("excluded"),
                excluded_total=retrieval.get("excluded_total"),
            )

        if not documents:
            fallback = (
                "Nisam pronašao relevantne informacije u bazi znanja "
                "za ovo pitanje."
            )
            yield self._to_script(fallback, script)
            return

        # ── Step 3: Build messages — question + chunks, no history ────── #
        context  = self._build_context(documents, metadatas)
        messages = self._build_messages(context, search_question)
        logger.info("Answering %r with %d chunk(s)", search_question, len(documents))
        self._record_messages(trace, messages)

        # ── Step 4: Stream from /api/chat ─────────────────────────────── #
        # Tokens pass through the normaliser; the last, possibly incomplete
        # word is held back so it is corrected as a whole and flushed after
        # the model's stream ends.
        normalizer = self._answer_normalizer(documents, script)
        answer: list[str] = []
        with self._stage(trace, "answer"):
            for token in self.llm_manager.chat_stream(messages):
                ready = normalizer.feed(token)
                if ready:
                    answer.append(ready)
                    yield ready
        tail = normalizer.flush()
        if tail:
            answer.append(tail)
            yield tail

        # ── Step 5: Append sources block ──────────────────────────────── #
        sources = self._answer_sources("".join(answer), metadatas, script)
        if trace is not None:
            trace["sources"] = sources
        if sources:
            # The "Izvori:" marker stays Latin: it is the API/UI contract
            # (web_ui/lib/models/conversation.dart), not model text.
            yield "\n\n---\nIzvori:\n" + "\n".join(sources)
