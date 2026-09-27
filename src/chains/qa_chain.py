"""
qa_chain.py
~~~~~~~~~~~
University QA system — retrieves relevant chunks, formats citations,
and generates a grounded answer via the local LLM.

Retrieval
---------
Hybrid dense (E5) + BM25 retrieval fused with Reciprocal Rank Fusion,
with an optional cross-encoder rerank stage (see ``src/chains/retriever.py``).

Citations
---------
A single :class:`CitationRegistry` assigns ``[Izvor N]`` numbers to
deduplicated ``(source, page)`` pairs.  The very same registry is used to
tag the context chunks *and* to render the source list, so a citation can
never renumber between the two.  After streaming, ``[Izvor N]`` markers
are stripped from the answer text (the client renders citations from the
structured ``sources`` event).  Only sources the model actually cited are
emitted; when a grounded answer carries no markers, it is attributed back
to the chunks it visibly draws from.  Refusals suppress the source list
entirely.

Multi-turn conversation support
------------------------------
- Maintains per-session conversation history (owned by the API layer).
- History is used for exactly one purpose: deciding whether the current
  request can stand on its own and, when it cannot, rewriting it into a
  standalone search query.  That is the single "operation A" LLM call in
  ``_plan_query`` — it receives a bounded transcript of the conversation
  and returns the current question unchanged when it is already
  self-contained.  No keyword/regex classification is used: the semantic
  relationship between the request and the conversation is judged by the
  model.
- The final answer ("operation B") never sees the conversation history.
  The prompt is always exactly two messages: the system prompt and one user
  message holding the retrieved context plus the *resolved* (standalone)
  question.  This keeps every answer anchored to a single topic, prevents
  earlier answers from contaminating the current one, and keeps the prompt
  bounded (context is capped per-chunk and in total).
- Follow-ups asking for *additional* information about the previous answer
  retrieve deeper (larger k) and exclude the chunks that already produced
  that answer, so the model expands with new facts instead of repeating.
- When the operation-A call fails (timeout, empty or drifting output) a
  deterministic fallback builds the best query it can from the previous
  clear question; a request that cannot be resolved is still retrieved
  with its original wording and the answer model may refuse when the
  context contains nothing relevant.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from src.chains.retriever import HybridRetriever, RetrievedChunk

logger = logging.getLogger(__name__)


# Fixed, deterministic refusal sentence — the prompt demands exactly this
# wording and the post-processing recognises it to suppress sources.
REFUSAL_PHRASE = (
    "Nažalost, nemam informaciju o tome u dostupnim izvorima."
)


def _normalize_for_match(text: str) -> str:
    """Lower-case, strip Serbian diacritics and collapse whitespace."""
    text = text.lower().replace("đ", "dj")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()


def _same_term(word: str, other: str) -> bool:
    """Inflection-tolerant word match.

    Two words are the same term when they share a 4-character prefix
    (``student``/``studentima``) or one is the prefix of the other
    (``kad``/``kada``), which absorbs the Serbian case endings the
    heuristics here rely on.
    """
    return (
        word[:4] == other[:4]
        or other.startswith(word)
        or word.startswith(other)
    )


# ── answer → source attribution (fallback for uncited answers) ──────── #
# The model is asked to tag facts with [Izvor N], but smaller local models
# sometimes answer from the context without any marker.  In that case the
# answer is matched back to the chunks it visibly draws from, so the client
# still shows the relevant sources instead of all retrieved ones.
_ATTRIBUTION_MIN_WORD_LEN = 5
# A chunk must score at least this fraction of the best-matching chunk.
_ATTRIBUTION_RELATIVE_THRESHOLD = 0.5
_ATTRIBUTION_MAX_SOURCES = 3


def _content_terms(text: str) -> set[str]:
    """Normalized words long enough to carry topical meaning."""
    words = re.findall(r"[a-z0-9]+", _normalize_for_match(text))
    return {word for word in words if len(word) >= _ATTRIBUTION_MIN_WORD_LEN}


def _attribute_chunks(
    answer: str, chunks: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    """
    Deterministically match an uncited answer back to the context chunks.

    Each chunk is scored by the words it shares with the answer, weighted
    by inverse document frequency across the retrieved chunks.  IDF
    suppresses generic corpus vocabulary ("master", "studenti"), so a
    chunk only wins on the terms that make the answer specific.  Chunks
    well below the best match are dropped, which keeps unrelated documents
    out of the source list.

    Returns the selected chunks (at most ``_ATTRIBUTION_MAX_SOURCES``, one
    per source page).
    """
    answer_terms = _content_terms(answer)
    if not answer_terms or not chunks:
        return []

    chunk_terms = [_content_terms(chunk.document) for chunk in chunks]
    document_frequency: dict[str, int] = {}
    for terms in chunk_terms:
        for word in terms & answer_terms:
            document_frequency[word] = document_frequency.get(word, 0) + 1

    total = len(chunks)
    scored: list[tuple[float, RetrievedChunk]] = []
    for chunk, terms in zip(chunks, chunk_terms):
        shared = terms & answer_terms
        if not shared:
            continue
        score = sum(
            math.log(1.0 + total / document_frequency[word])
            for word in shared
        )
        scored.append((score, chunk))

    if not scored:
        return []

    scored.sort(key=lambda item: item[0], reverse=True)
    best = scored[0][0]
    selected: list[RetrievedChunk] = []
    seen: set[tuple] = set()
    for score, chunk in scored:
        if score < best * _ATTRIBUTION_RELATIVE_THRESHOLD:
            break
        key = (chunk.metadata.get("source"), chunk.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        selected.append(chunk)
        if len(selected) >= _ATTRIBUTION_MAX_SOURCES:
            break
    return selected


_LEGACY_REFUSAL_MARKERS = (
    "nisam pronasao relevantne informacije",
    "nisam pronašao relevantne informacije",
)

# Refusals start with one of these; a long grounded answer that merely ends
# with a stray refusal sentence must not be classified as a refusal.
_REFUSAL_PREFIXES = (
    _normalize_for_match(REFUSAL_PHRASE),
    "nemam informaciju",
    "nazalost, nemam",
    "nisam pronasao",
    "nisam pronašao",
)

_REFUSAL_MARKERS = (
    "nemam informaciju",
) + _LEGACY_REFUSAL_MARKERS

# A trailing "nemam informaciju..." sentence tacked onto a real answer.
_STRAY_REFUSAL_RE = re.compile(
    r"(?is)[ \t]*(?:\n+[ \t]*)?"
    r"(?:(?:na osnovu (?:dostupnog )?konteksta|nažalost|nazalost)[,\s]*)?"
    r"nemam informaciju[^.!?\n]{0,80}[.!?][ \t]*$"
)


def _strip_stray_refusal_sentence(answer: str, was_grounded: bool = False) -> str:
    """Drop a final "nemam informaciju..." sentence when a real, grounded
    answer precedes it.  A pure refusal is returned unchanged.

    ``was_grounded`` is set by callers that already know the model used at
    least one valid source this turn; it compensates for the citation
    markers having been stripped from the answer text.
    """
    match = _STRAY_REFUSAL_RE.search(answer)
    if not match:
        return answer
    candidate = answer[: match.start()].strip()
    if was_grounded or (
        len(candidate) >= 80 and _CITATION_REGEX.search(candidate)
    ):
        return candidate
    return answer


def _is_refusal(answer: str, was_grounded: bool = False) -> bool:
    """True when the model refused to answer from the given context.

    Only an answer that *starts* with the fixed refusal (or a short variant
    of it) counts — a long grounded answer ending with a stray refusal
    sentence keeps its sources.
    """
    normalized = _normalize_for_match(answer)
    if any(normalized.startswith(prefix) for prefix in _REFUSAL_PREFIXES):
        return True
    if normalized.startswith("na osnovu konteksta") and any(
        marker in normalized[:100] for marker in _REFUSAL_MARKERS
    ):
        # "Na osnovu konteksta, nemam informaciju..." is a refusal, while
        # "Na osnovu konteksta, predmet nosi..." is a grounded answer.
        return True

    # Ignore a stray refusal sentence appended to an otherwise grounded
    # answer ("... [Izvor 2]. Na osnovu konteksta, nemam informaciju...").
    stripped = _normalize_for_match(
        _strip_stray_refusal_sentence(answer, was_grounded)
    )
    return len(stripped) <= 120 and any(
        marker in stripped for marker in _REFUSAL_MARKERS
    )


# ── citation parsing / validation ───────────────────────────────────── #
# Matches [Izvor 1], [izvor 2], [Izvor 1, 3], [Izvor 1 i 2].
_CITATION_REGEX = re.compile(
    r"\[\s*izvor\s*(\d+(?:\s*(?:,|i)\s*\d+)*)\s*\]",
    re.IGNORECASE,
)
_NUMBERS_REGEX = re.compile(r"\d+")


def _clean_spacing(text: str) -> str:
    """Tidy the whitespace left behind when a citation is removed."""
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def _filter_citations(text: str, valid_numbers: set[int]) -> tuple[str, set[int]]:
    """
    Remove every ``[Izvor N]`` marker from ``text`` and return the numbers
    the model cited that actually exist in ``valid_numbers``.

    Markers are stripped because citations are rendered by the client from
    the structured ``sources`` payload, not from the answer text.
    """
    cited: set[int] = set()

    def replace(match: re.Match) -> str:
        numbers = [int(n) for n in _NUMBERS_REGEX.findall(match.group(1))]
        cited.update(n for n in numbers if n in valid_numbers)
        return ""

    cleaned = _CITATION_REGEX.sub(replace, text)
    return _clean_spacing(cleaned), cited


class _CitationStreamFilter:
    """
    Removes ``[Izvor N]`` markers while tokens stream and records which
    citation numbers the model used.

    The answer text sent to the client carries no citation markers — the
    client renders the ``sources`` payload instead — but the numbers are
    still needed to decide which sources to emit.  Partial markers are
    held back until their closing bracket arrives, so a marker split
    across tokens is never leaked.  Trailing whitespace is held back just
    long enough to be dropped together with a marker that follows it,
    which avoids leftover spaces before punctuation.  A marker that is
    obviously not a citation (e.g. a Markdown link) is released
    immediately.
    """

    _PARTIAL_CITATION_RE = re.compile(
        r"^\[\s*(?:i(?:z(?:v(?:o(?:r)?)?)?)?)?\s*(?:\d+\s*(?:[,i]\s*\d*\s*)*)?$",
        re.IGNORECASE,
    )
    _MAX_MARKER_LEN = 48

    def __init__(self, valid_numbers: set[int]) -> None:
        self._valid = set(valid_numbers)
        self._pending = ""
        self._whitespace = ""
        self.cited: set[int] = set()

    def feed(self, token: str) -> str:
        self._pending += token
        return self._drain(final=False)

    def finish(self) -> str:
        out = self._drain(final=True)
        self._pending = ""
        return out

    def _sanitize(self, marker: str) -> str:
        match = _CITATION_REGEX.fullmatch(marker)
        if not match:
            return marker
        numbers = [int(n) for n in _NUMBERS_REGEX.findall(match.group(1))]
        self.cited.update(n for n in numbers if n in self._valid)
        return ""

    def _drain(self, final: bool) -> str:
        out: list[str] = []
        while self._pending:
            bracket = self._pending.find("[")
            if bracket == -1:
                out.append(self._pending)
                self._pending = ""
                break

            out.append(self._pending[:bracket])
            rest = self._pending[bracket:]

            closing = rest.find("]")
            if closing != -1:
                out.append(self._sanitize(rest[: closing + 1]))
                self._pending = rest[closing + 1 :]
                continue

            # No closing bracket yet.  Hold the tail only while it can still
            # become a citation; otherwise release it.
            if (
                not final
                and len(rest) <= self._MAX_MARKER_LEN
                and self._PARTIAL_CITATION_RE.match(rest)
            ):
                self._pending = rest
                break

            out.append(rest[0])
            self._pending = rest[1:]

        return self._release("".join(out), final)

    def _release(self, text: str, final: bool) -> str:
        """Tidy the whitespace around dropped citations.

        A space that preceded a removed marker must not survive in front of
        the punctuation that follows it ("ESPB [Izvor 2]." → "ESPB.").
        Trailing whitespace is held back so the same holds when the marker
        and the punctuation arrive in different tokens.
        """
        combined = self._whitespace + text
        self._whitespace = ""
        combined = re.sub(r"[ \t]+([,.;:!?])", r"\1", combined)
        if final:
            return combined

        # Avoid a double space when the previous token ended with one that
        # was held back while a marker resolved.
        combined = re.sub(r"^[ \t]{2,}", " ", combined)
        stripped = combined.rstrip()
        if not stripped:
            self._whitespace = combined
            return ""
        self._whitespace = combined[len(stripped) :]
        return stripped


# ── streaming events ────────────────────────────────────────────────── #
@dataclass(frozen=True)
class TokenChunk:
    """A piece of the answer, with citation markers already stripped."""

    text: str


@dataclass(frozen=True)
class SourcesChunk:
    """Structured source records to be sent as a dedicated SSE event."""

    sources: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievedIdsChunk:
    """Ids of the chunks used for this turn's context.

    The API layer stores them in the session so that a follow-up asking for
    *additional* information can exclude the already-seen chunks and retrieve
    genuinely new material instead of the same top-k results.
    """

    chunk_ids: list[str] = field(default_factory=list)


# ── "give me more information about the previous answer" follow-ups ──── #
# Questions matching these ask for *additional* details about a topic
# already discussed — retrieval must go deeper (larger k) and skip the
# chunks that already produced the previous answer.
_MORE_INFO_PATTERNS = [
    r"više informacij\w*", r"vise informacij\w*",
    r"još informacij\w*", r"jos informacij\w*",
    r"više o\b", r"vise o\b",
    r"detaljnij\w*", r"više detalj\w*", r"vise detalj\w*",
    r"još detalj\w*", r"jos detalj\w*",
    r"dodatn\w* informacij\w*", r"dodatn\w* detalj\w*",
    r"nabroj\w*", r"nabroji još\b", r"nabroji jos\b",
    r"navedi još\b", r"navedi jos\b",
    r"šta još\b", r"sta jos\b", r"još nešto\b", r"jos nesto\b",
    r"još nečega\b", r"jos necega\b", r"nešto još\b", r"nesto jos\b",
    r"išta još\b", r"ista jos\b",
    r"reci više\b", r"reci vise\b", r"pričaj više\b", r"pricaj vise\b",
    r"ima li još\b", r"ima li jos\b", r"imas li još\b", r"imas li jos\b",
    r"znas li još\b", r"znas li jos\b", r"da li ima još\b", r"da li ima jos\b",
    r"koji još\b", r"koji jos\b", r"koje još\b", r"koje jos\b",
    r"kakve još\b", r"kakve jos\b", r"koliko još\b", r"koliko jos\b",
    r"objasni\w*", r"pojasni\w*",
    r"more information", r"tell me more", r"\bwhat else\b",
    r"more details\b", r"\bmore about\b", r"elaborate\w*", r"explain\w*",
]

_MORE_INFO_REGEX = re.compile(
    r"(?i)" + "|".join(f"(?:{p})" for p in _MORE_INFO_PATTERNS)
)

# Generic phrases that carry no topic of their own, used to detect a
# follow-up that is only "tell me more about that/them".
_FOLLOWUP_STOPWORDS = {
    "a", "i", "pa", "ali", "o", "na", "u", "za", "sa", "od", "do", "iz",
    "je", "su", "da", "li", "mi", "nam", "se", "bi", "bih",
    "ko", "sta", "koji", "koje", "koja", "kakav", "kakva", "kakve",
    "kako", "zasto", "gde", "gdje", "koliko",
    "mozes", "moze", "mozemo", "mogu", "moci", "kazes", "kaze",
    "reci", "znas", "znate", "objasni", "pojasni", "elaboriraj",
    "detaljnije", "detalje", "detalja", "detaljno", "detalji",
    "detaljima", "opširnije", "opsirnije",
    "vise", "jos", "informacije", "informacija", "podataka",
    "njih", "njima", "njega", "njemu", "njihov", "ih", "ga", "mu",
    "tog", "tom", "tim", "to", "ovo", "ovom", "ovog", "ove",
    "tome", "toga", "tih", "tima", "taj", "ta", "tu",
    "nesto", "neki", "neka", "vezano", "vezi",
    "necega", "necemu", "ista", "isto",
    "ima", "imas", "imam", "imamo", "imate", "imaju",
    "daj", "dajte", "kazi", "napisi", "navedi", "pricaj",
    "izvor", "izvori", "izvora", "izvoru",
    "please", "tell", "more", "about", "them", "that",
}


@dataclass(frozen=True)
class _QueryPlan:
    """Outcome of the per-question decision flow.

    ``query`` is the standalone retrieval query to embed and search — the
    original question when it already stands on its own, or the rewritten
    version when the operation-A model resolved it from the conversation.
    ``reason`` is a short label for the decision, used in logs.
    """

    query: str
    reason: str


class CitationRegistry:
    """
    Maps deduplicated ``(source, page)`` pairs to stable ``[Izvor N]``
    numbers for the duration of a single query.

    The same numbers are used for context tags and for the rendered source
    list, which guarantees that ``[Izvor N]`` never renumbers.
    """

    def __init__(self) -> None:
        self._numbers: dict[tuple[str, int], int] = {}
        self._metadatas: list[dict] = []

    @staticmethod
    def _key(meta: dict) -> tuple[str, int]:
        source = str(meta.get("source") or meta.get("filename") or "")
        page = int(meta.get("page") or 0)
        return source, page

    def add(self, meta: dict) -> int:
        """Register a chunk's metadata and return its citation number."""
        key = self._key(meta)
        if key not in self._numbers:
            self._numbers[key] = len(self._metadatas) + 1
            self._metadatas.append(meta)
        return self._numbers[key]

    def number_for(self, meta: dict) -> int:
        return self.add(meta)

    @property
    def valid_numbers(self) -> set[int]:
        return set(range(1, len(self._metadatas) + 1))

    @staticmethod
    def entry(meta: dict, number: int) -> dict:
        filename = meta.get("filename") or meta.get("source") or "Nepoznat izvor"
        page = int(meta.get("page") or 0)
        url = meta.get("url") or ""
        doc_type = meta.get("type") or ""

        if doc_type == "pdf" and page:
            label = f"{filename} — strana {page}"
        elif url:
            label = f"{filename} — {url}"
        else:
            label = filename

        if page:
            context_label = f"{filename}, strana {page}"
        elif url:
            context_label = f"{filename} ({url})"
        else:
            context_label = filename

        return {
            "number": number,
            "filename": filename,
            "page": page,
            "url": url,
            "type": doc_type,
            "label": label,
            "context_label": context_label,
        }

    def entries(self, cited: set[int] | None = None) -> list[dict]:
        rows = []
        for index, meta in enumerate(self._metadatas, start=1):
            if cited is not None and index not in cited:
                continue
            rows.append(self.entry(meta, index))
        return rows

    def footer(self, cited: set[int] | None = None) -> list[str]:
        return [f"[{row['number']}] {row['label']}" for row in self.entries(cited)]

    def structured(self, cited: set[int] | None = None) -> list[dict]:
        return [
            {
                "number": row["number"],
                "filename": row["filename"],
                "page": row["page"],
                "url": row["url"],
                "type": row["type"],
                "label": row["label"],
            }
            for row in self.entries(cited)
        ]


class UniversityQASystem:
    # System prompt sent as the first message in every /api/chat call.
    # Hardened: answer only from context, no invented facts, canonical
    # citations, fixed refusal phrase.
    SYSTEM_PROMPT = (
        "Vi ste univerzitetski asistent Instituta za matematiku i informatiku. "
        "Odgovarajte isključivo na srpskom latiničnom pismu.\n\n"
        "PRAVILA:\n"
        "1. Odgovorite SAMO na osnovu informacija iz priloženog KONTEKSTA. "
        "Ne koristite spoljno znanje i ne izmišljajte činjenice, brojeve, "
        "datume, imena ni izvore.\n"
        "2. Informacije mogu biti u rečenicama, listama, tabelama i naslovima — "
        "iskoristite ih ako odgovaraju na pitanje. Ako pitanje traži dodatne "
        "informacije ili pojašnjenje, navedite konkretne detalje iz konteksta "
        "(liste, nazive, uslove, predmete).\n"
        "3. Ako odgovor nije moguće pronaći u kontekstu, odgovorite tačno "
        f'ovom rečenicom i ništa više: "{REFUSAL_PHRASE}"\n'
        "4. Uz svaku činjenicu koju preuzimate iz konteksta obavezno navedite "
        "izvor tačno u obliku [Izvor N], gde je N broj iz konteksta; svaka "
        "rečenica sa činjenicom mora sadržati bar jedan [Izvor N]. "
        'Primer: "Predmet Uvod u programiranje nosi 6 ESPB bodova [Izvor 2]."\n'
        "5. Nikada ne navodite broj izvora koji ne postoji u kontekstu i ne "
        "izmišljajte izvore.\n"
        "6. Ne pominjite da ste jezički model, ne tražite dodatne informacije "
        "od korisnika i ne dodajte opšte savete van konteksta.\n"
        "7. Odgovor neka bude KRATAK i direktan: najviše 3–5 rečenica koje "
        "odgovaraju na pitanje, bez ponavljanja pitanja, bez uvodnih fraza i "
        "bez prepisivanja celih odlomaka iz konteksta."
    )

    def __init__(self, llm_manager, vector_store, translator):
        self.llm_manager  = llm_manager
        self.vector_store = vector_store
        self.translator   = translator

        config = getattr(vector_store, "config", None) or getattr(
            llm_manager, "config", None
        )
        self.config = config
        self.retriever = HybridRetriever(vector_store, config)

        self.default_k = int(getattr(config, "TOP_K_RESULTS", 5))
        # Follow-ups that ask for additional information about the previous
        # answer retrieve deeper so new chunks (not the ones that produced
        # that answer) reach the context.
        self.more_info_k = int(getattr(config, "MORE_INFO_TOP_K", 10))
        self.rewrite_timeout = getattr(config, "REWRITE_TIMEOUT_SECONDS", 6.0)

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
        return f"[{index}] {CitationRegistry.entry(meta, index)['label']}"

    def _format_sources(self, metadatas: list[dict]) -> list[str]:
        """
        Deduplicate and format all sources for a query result.
        Deduplication key: (source_path_or_url, page) so we don't list
        the same file page twice when two chunks came from the same page.
        """
        registry = CitationRegistry()
        for meta in metadatas:
            registry.add(meta)
        return registry.footer()

    # ------------------------------------------------------------------ #
    #  Context building                                                    #
    # ------------------------------------------------------------------ #

    # Bounded context: long chunks are truncated and the total context is
    # capped, so deep retrieval can never blow up the prompt.
    _CONTEXT_CHUNK_CHAR_LIMIT = 800
    _CONTEXT_TOTAL_CHAR_LIMIT = 6000

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk], registry: CitationRegistry) -> str:
        """
        Interleave each chunk with its registry citation tag so the LLM can
        reference the exact source number that the footer will use.

        Example:
            [Izvor 1 — lecture_notes.pdf, strana 4]
            <chunk text here>
        """
        parts: list[str] = []
        total = 0
        for chunk in chunks:
            number = registry.number_for(chunk.metadata)
            row = registry.entry(chunk.metadata, number)
            text = chunk.document
            if len(text) > UniversityQASystem._CONTEXT_CHUNK_CHAR_LIMIT:
                text = text[: UniversityQASystem._CONTEXT_CHUNK_CHAR_LIMIT] + "…"
            block = f"[Izvor {number} — {row['context_label']}]\n{text}"
            if parts and total + len(block) > UniversityQASystem._CONTEXT_TOTAL_CHAR_LIMIT:
                break
            parts.append(block)
            total += len(block)

        logger.info(
            "Context: %d of %d retrieved chunk(s) passed to the LLM",
            len(parts),
            len(chunks),
        )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    #  Conversation history helpers                                        #
    # ------------------------------------------------------------------ #

    # ── Operation A input: a bounded transcript of the conversation ──── #
    # The follow-up/rewrite call receives the *whole* recent conversation
    # (not just the last answer), so a reference can be resolved against an
    # entity named several turns back.  Message count, per-message length
    # and total length are capped so the rewrite prompt stays small even in
    # long conversations.
    _REWRITE_HISTORY_MAX_MESSAGES = 6
    _REWRITE_USER_CHAR_LIMIT = 300
    _REWRITE_ASSISTANT_CHAR_LIMIT = 600
    _REWRITE_HISTORY_CHAR_LIMIT = 3200

    @classmethod
    def _format_history_for_rewrite(cls, history: list[dict[str, str]]) -> str:
        """
        Render the recent conversation as a compact ``role: content`` block
        for the follow-up detection / query rewriting call.

        The last few turns are kept (a follow-up can refer to an entity
        named several messages back) while each message is truncated and
        the total length capped.
        """
        lines: list[str] = []
        for msg in history[-cls._REWRITE_HISTORY_MAX_MESSAGES:]:
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if msg.get("role") == "user":
                role, limit = "Korisnik", cls._REWRITE_USER_CHAR_LIMIT
            else:
                role, limit = "Asistent", cls._REWRITE_ASSISTANT_CHAR_LIMIT
            if len(content) > limit:
                content = content[:limit] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)[-cls._REWRITE_HISTORY_CHAR_LIMIT:]

    @staticmethod
    def _same_query(left: str, right: str) -> bool:
        """True when two queries are identical apart from case/diacritics."""
        return _normalize_for_match(left) == _normalize_for_match(right)

    @staticmethod
    def _is_more_info_followup(question: str) -> bool:
        """
        True when the question asks for *additional* information about a
        topic from the previous answer ("daj mi više informacija o njima",
        "šta još?", "nabroji još predmeta", "tell me more").
        """
        return bool(_MORE_INFO_REGEX.search(question))

    @staticmethod
    def _previous_user_question(question: str, history: list[dict[str, str]]) -> str:
        """Return the most recent user question that is not the current one.

        Referential questions ("a šta je sa njima?") are skipped when an
        older, self-contained question exists: they carry no topic of their
        own and would retrieve badly.  This keeps longer follow-up chains
        anchored to the last question that actually names a subject.  Used
        only to build a deterministic fallback when the model rewrite is
        unavailable.
        """
        current = question.strip()
        candidates: list[str] = []
        for msg in reversed(history):
            if msg.get("role") != "user":
                continue
            content = (msg.get("content") or "").strip()
            if content and content != current:
                candidates.append(content)

        for content in candidates:
            if UniversityQASystem._has_topic_words(content):
                return content
        return candidates[0] if candidates else ""

    @staticmethod
    def _content_words(text: str) -> set[str]:
        """Normalized, non-generic words — the topical anchors of a text."""
        words = re.findall(r"[a-z0-9]+", _normalize_for_match(text))
        return {w for w in words if len(w) > 2 and w not in _FOLLOWUP_STOPWORDS}

    @staticmethod
    def _has_topic_words(question: str) -> bool:
        """True when the question contains something more than generic
        follow-up phrasing ("what can you tell me about them")."""
        return bool(UniversityQASystem._content_words(question))

    def _rewrite_keeps_topic(
        self, rewritten: str, question: str, history: list[dict[str, str]]
    ) -> bool:
        """Reject a rewrite that drifted away from the conversation topic.

        Two conditions must hold.  First, the rewrite must keep the topic of
        the latest question: when the question has content words, at least
        one of them must survive in the rewrite (compared by a 4-character
        prefix, which absorbs Serbian inflection).  This is what stops a
        rewrite from silently swapping the named subject for the topic of
        the previous answer ("a koliko ima na informatici?" must not become
        a question about matematika).  Second, the rewrite must share at
        least one content word with the current question, the previous clear
        question or the last assistant answer.  A rewrite that fails either
        check is treated as a hallucination and the deterministic fallback
        is used instead.
        """
        rewrite_words = self._content_words(rewritten)
        if not rewrite_words:
            return False

        question_words = self._content_words(question)
        if question_words and not any(
            _same_term(word, rewrite_word)
            for word in question_words
            for rewrite_word in rewrite_words
        ):
            return False

        anchors = self._content_words(question)
        previous = self._previous_user_question(question, history)
        if previous:
            anchors |= self._content_words(previous)
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                anchors |= self._content_words(msg.get("content", ""))
                break

        if not anchors:
            return True
        return any(
            _same_term(word, anchor)
            for word in rewrite_words
            for anchor in anchors
        )

    def _context_fallback_query(
        self, question: str, history: list[dict[str, str]]
    ) -> str:
        """
        Deterministic rewrite used when the LLM rewrite fails, times out or
        gives up.

        A purely referential follow-up ("Šta možeš o njima?") retrieves with
        the previous question alone, while a follow-up with its own topic
        words ("A koliko ESPB ima softversko inženjerstvo?") is prepended
        with the previous question so both topics contribute.  Either way
        this stays a single query — no extra embedding calls.
        """
        previous = self._previous_user_question(question, history)
        if not previous:
            return question

        if not self._has_topic_words(question):
            logger.info(
                "Follow-up has no topic words — retrieving with previous question only"
            )
            return previous

        combined = f"{previous} {question}".strip()
        logger.info("Using previous question + current query: %r", combined[:160])
        return combined

    # ------------------------------------------------------------------ #
    #  Query contextualization                                             #
    # ------------------------------------------------------------------ #

    # Labels/quotes the rewrite model sometimes wraps its output in.
    _REWRITE_LABEL_RE = re.compile(
        r"^\s*(?:prepisano pitanje|samostalno pitanje|prepisano|pitanje"
        r"|odgovor)\s*[:：\-]\s*",
        re.IGNORECASE,
    )

    @staticmethod
    def _clean_rewrite_output(text: str) -> str:
        """Strip surrounding quotes and leading labels from a rewrite."""
        text = text.strip().strip('"\'`“”„”«»')
        text = UniversityQASystem._REWRITE_LABEL_RE.sub("", text, count=1)
        return text.strip().strip('"\'`“”„”«»')

    def _try_llm_rewrite(
        self, question: str, history: list[dict[str, str]]
    ) -> str | None:
        """
        Operation A: one bounded LLM call that decides whether the current
        request can stand on its own and, when it cannot, rewrites it into
        a standalone query using the previous conversation.

        The model receives a bounded transcript of the conversation, so a
        reference can be resolved against an entity named several turns
        back.  Returns ``None`` when the call fails, times out, produces an
        empty or implausibly long answer, or drifts away from the
        conversation topic — the caller then uses the deterministic
        fallback instead.
        """
        history_text = self._format_history_for_rewrite(history)
        if not history_text:
            return None

        try:
            rewritten = self.llm_manager.rewrite_query(
                history_text,
                question,
                timeout=self.rewrite_timeout,
            )
        except Exception:
            logger.exception("Query rewrite failed — using deterministic fallback")
            return None

        rewritten = self._clean_rewrite_output(rewritten or "")
        if not rewritten:
            logger.info("Rewrite unavailable — using deterministic fallback")
            return None
        max_chars = max(len(question) * 5, 200)
        if len(rewritten) < len(question) * 0.3 or len(rewritten) > max_chars:
            logger.warning(
                "Rewrite length implausible (%d → %d chars) — using fallback",
                len(question),
                len(rewritten),
            )
            return None
        if not self._rewrite_keeps_topic(rewritten, question, history):
            logger.warning("Rewrite drifted from the conversation topic")
            return None

        logger.info("Query rewrite: %r → %r", question, rewritten)
        return rewritten

    def _plan_query(
        self, question: str, history: list[dict[str, str]]
    ) -> _QueryPlan:
        """
        Decision flow executed once per incoming request.  It decides which
        query is embedded, searched and finally put in front of the answer
        model:

        1. No conversation history → the request is standalone by
           definition; it is searched as-is and no rewrite call is made.
        2. History present → one semantic operation-A call receives the
           bounded conversation and the current question and decides
           whether the request is understandable on its own:
             a. model returns the question unchanged → standalone;
             b. model returns a self-contained rewrite → rewritten query;
             c. call failed / output unusable      → deterministic fallback
                built from the previous clear question.
        The raw conversation history never reaches the answer model.
        """
        if not history:
            logger.info("Request flow: no history — treating request as standalone")
            return _QueryPlan(question, "standalone")

        logger.info("Request flow: history present — checking for follow-up")
        rewritten = self._try_llm_rewrite(question, history)
        if rewritten is None:
            fallback = self._context_fallback_query(question, history)
            return _QueryPlan(fallback, "fallback")
        if self._same_query(rewritten, question):
            return _QueryPlan(question, "standalone")
        return _QueryPlan(rewritten, "rewritten")

    def _standalone_question(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> str:
        """
        Resolve the latest user question into a standalone retrieval query.

        Thin wrapper over :meth:`_plan_query` for callers that only need
        the query text.
        """
        return self._plan_query(question, history or []).query

    # ------------------------------------------------------------------ #
    #  Message list builder                                                #
    # ------------------------------------------------------------------ #

    # Fixed instruction appended to the user prompt when the question asks
    # for additional information (retrieval already replaced the previously
    # used chunks with fresh ones).
    _MORE_INFO_NOTE = (
        "NAPOMENA: Korisnik traži dodatne informacije. Iz konteksta iznesite "
        "konkretne činjenice, liste ili detalje i za svaku navedite [Izvor N]."
    )

    def _build_user_content(
        self, context: str, question: str, more_info: bool = False
    ) -> str:
        """Wrap the current question with retrieved context for the LLM."""
        note = f"{self._MORE_INFO_NOTE}\n\n" if more_info else ""
        return (
            f"Kontekst:\n{context}\n\n"
            f"Pitanje: {question}\n\n"
            f"{note}"
            "Odgovor neka bude kratak i direktan, bez uvodnih fraza. "
            "Odgovorite isključivo na osnovu konteksta i navedite [Izvor N] "
            "za svaku činjenicu. Ako odgovora nema u kontekstu, napišite "
            f'tačno: "{REFUSAL_PHRASE}"'
        )

    # ------------------------------------------------------------------ #
    #  Retrieval with fallback                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dedupe_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Drop duplicate chunks before they reach the context window."""
        seen_ids: set[str] = set()
        seen_text: set[str] = set()
        unique: list[RetrievedChunk] = []
        for chunk in chunks:
            text_key = re.sub(r"\s+", " ", chunk.document).strip().lower()
            if chunk.id in seen_ids or text_key in seen_text:
                continue
            seen_ids.add(chunk.id)
            seen_text.add(text_key)
            unique.append(chunk)
        return unique

    def _search(self, query: str, k: int) -> list[RetrievedChunk]:
        normalized = self.translator.normalize_query(query)
        return self._dedupe_chunks(self.retriever.retrieve(normalized, k=k))

    def _retrieve(
        self,
        plan: _QueryPlan,
        k: int | None = None,
        more_info: bool = False,
        exclude_chunk_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Single retrieval pass over the query chosen by the decision flow.

        A follow-up that asks for additional information about the previous
        answer ("daj mi više informacija o njima", "šta još?") retrieves
        deeper (``more_info_k``) and drops the chunks that already produced
        the previous answer, so the model gets genuinely new material to
        expand on instead of the same top-k chunks.

        The query is the standalone one chosen by ``_plan_query``.  The
        existing retriever (dense similarity threshold, hybrid/RRF fusion
        and optional cross-encoder rerank) is the chunk-selection stage, so
        unrelated chunks are not blindly handed to the answer model.
        """
        k = k or self.default_k
        search_k = max(k, self.more_info_k) if more_info else k
        exclude = set(exclude_chunk_ids or [])

        chunks = self._search(plan.query, search_k)

        if more_info and exclude:
            fresh = [c for c in chunks if c.id not in exclude]
            if len(fresh) >= max(2, k // 2):
                logger.info(
                    "More-info follow-up: %d already-seen chunk(s) excluded, "
                    "%d fresh chunk(s) kept",
                    len(chunks) - len(fresh),
                    len(fresh),
                )
                chunks = fresh
            else:
                logger.info(
                    "Only %d fresh chunk(s) after exclusion — keeping all retrieved",
                    len(fresh),
                )

        if chunks:
            logger.info("Retrieval OK (%d chunks) for %r", len(chunks), plan.query)
        else:
            logger.warning("Retrieval returned no documents for %r", plan.query)
        return chunks

    # ------------------------------------------------------------------ #
    #  Prompt preparation                                                  #
    # ------------------------------------------------------------------ #

    def _prepare(
        self,
        question: str,
        history: list[dict[str, str]],
        k: int | None = None,
        exclude_chunk_ids: list[str] | None = None,
    ) -> tuple[CitationRegistry, list[dict[str, str]], list[RetrievedChunk]]:
        """
        Resolve the question, retrieve this turn's chunks and build the
        prompt.

        The prompt is always exactly two messages — the system prompt and
        one user message holding the retrieved context plus the *resolved*
        question.  Previous turns are deliberately not included: history is
        used only to rewrite an ambiguous question into a standalone one, so
        the model answers the current topic without mixing in earlier
        conversations' facts.  The decision flow runs once here and its
        query is shared by retrieval and the prompt.
        """
        plan = self._plan_query(question, history)
        more_info = bool(history) and self._is_more_info_followup(question)
        logger.info(
            "Query plan: original=%r follow_up=%s reason=%s search_query=%r",
            question,
            plan.reason != "standalone",
            plan.reason,
            plan.query,
        )
        chunks = self._retrieve(
            plan,
            k=k,
            more_info=more_info,
            exclude_chunk_ids=exclude_chunk_ids,
        )
        registry = CitationRegistry()
        for chunk in chunks:
            registry.add(chunk.metadata)

        if not chunks:
            return registry, [], []

        context = self._build_context(chunks, registry)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_user_content(
                    context, plan.query, more_info
                ),
            },
        ]
        return registry, messages, chunks

    # ------------------------------------------------------------------ #
    #  Query  (non-streaming)                                              #
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        k: int | None = None,
        exclude_chunk_ids: list[str] | None = None,
    ) -> dict:
        """
        Retrieve relevant chunks, build a grounded prompt, call the LLM,
        and return the answer alongside formatted source references.

        Returns
        -------
        {
            "question": str,
            "answer":   str,
            "sources":  list[str],        # human-readable citation lines
            "sources_structured": list[dict],
            "chunk_ids": list[str],       # ids used this turn (for follow-ups)
        }
        """
        history = conversation_history or []
        registry, messages, chunks = self._prepare(
            question, history, k=k, exclude_chunk_ids=exclude_chunk_ids
        )

        if not chunks:
            response = self._empty_response(question)
            response["chunk_ids"] = []
            return response

        answer = self.llm_manager.chat(messages)
        answer, cited = self._finalize_answer(answer, registry)
        if not cited and not _is_refusal(answer):
            cited = self._attribute_cited(answer, chunks, registry)
        response = self._response(question, answer, registry, cited)
        response["chunk_ids"] = [c.id for c in chunks]
        return response

    def _empty_response(self, question: str) -> dict:
        return {
            "question": question,
            "answer": REFUSAL_PHRASE,
            "sources": [],
            "sources_structured": [],
        }

    @staticmethod
    def _attribute_cited(
        answer: str,
        chunks: list[RetrievedChunk],
        registry: CitationRegistry,
    ) -> set[int]:
        """Map an uncited answer back to the chunks it was grounded in."""
        return {
            registry.number_for(chunk.metadata)
            for chunk in _attribute_chunks(answer, chunks)
        }

    def _finalize_answer(
        self, answer: str, registry: CitationRegistry
    ) -> tuple[str, set[int]]:
        """Post-stream validation for the non-streaming path.

        Returns the citation-free answer text and the set of valid citation
        numbers the model used.
        """
        answer, cited = _filter_citations(answer, registry.valid_numbers)
        answer = _strip_stray_refusal_sentence(answer, was_grounded=bool(cited))
        return answer.strip(), cited

    def _response(
        self,
        question: str,
        answer: str,
        registry: CitationRegistry,
        cited: set[int],
    ) -> dict:
        if _is_refusal(answer, was_grounded=bool(cited)):
            return {
                "question": question,
                "answer": answer,
                "sources": [],
                "sources_structured": [],
            }
        return {
            "question": question,
            "answer": answer,
            "sources": registry.footer(cited),
            "sources_structured": registry.structured(cited),
        }

    # ------------------------------------------------------------------ #
    #  Query  (streaming)                                                  #
    # ------------------------------------------------------------------ #

    async def query_stream(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        k: int | None = None,
        exclude_chunk_ids: list[str] | None = None,
    ) -> AsyncGenerator[TokenChunk | SourcesChunk | RetrievedIdsChunk, None]:
        """
        Stream response tokens, then emit the structured source records.

        ``[Izvor N]`` markers are stripped from the streamed text (partial
        markers are buffered) while the cited numbers are recorded, so the
        client renders citations only from the ``sources`` payload.  On
        completion, a refusal suppresses the source list entirely.

        A ``RetrievedIdsChunk`` with the ids of the chunks used this turn is
        emitted first so the API layer can pass them back on the next turn
        (they are excluded for "more information" follow-ups).
        """
        history = conversation_history or []
        registry, messages, chunks = self._prepare(
            question, history, k=k, exclude_chunk_ids=exclude_chunk_ids
        )

        if not chunks:
            yield TokenChunk(REFUSAL_PHRASE)
            yield RetrievedIdsChunk([])
            yield SourcesChunk([])
            return

        yield RetrievedIdsChunk([c.id for c in chunks])

        stream_filter = _CitationStreamFilter(registry.valid_numbers)
        emitted: list[str] = []

        for token in self.llm_manager.chat_stream(messages):
            safe = stream_filter.feed(token)
            if safe:
                emitted.append(safe)
                yield TokenChunk(safe)

        tail = stream_filter.finish()
        if tail:
            emitted.append(tail)
            yield TokenChunk(tail)

        # ── Post-stream pass ──────────────────────────────────────────── #
        answer = _clean_spacing("".join(emitted))
        cited = set(stream_filter.cited)

        if _is_refusal(answer, was_grounded=bool(cited)):
            # Refusals carry no sources — the API must not render Izvori.
            yield SourcesChunk([])
        else:
            # Emit the sources the model cited, or — when it answered
            # without markers — the chunks the answer visibly draws from.
            # Never every retrieved chunk: that would surface irrelevant
            # documents and pile up as follow-ups retrieve deeper.
            if not cited:
                cited = self._attribute_cited(answer, chunks, registry)
            yield SourcesChunk(registry.structured(cited))
