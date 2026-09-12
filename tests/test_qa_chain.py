"""Unit tests for citation mapping, validation and refusal handling."""

import asyncio

from src.chains.qa_chain import (
    REFUSAL_PHRASE,
    CitationRegistry,
    SourcesChunk,
    TokenChunk,
    UniversityQASystem,
    _CitationStreamFilter,
    _filter_citations,
    _is_refusal,
    _strip_stray_refusal_sentence,
)
from src.chains.retriever import RetrievedChunk


def _chunk(document: str, source: str, page: int = 0, **extra) -> RetrievedChunk:
    metadata = {"source": source, "filename": source, "page": page, **extra}
    return RetrievedChunk(id=source, document=document, metadata=metadata)


class _FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query, k=5):
        return list(self._chunks)


class _CountingRetriever(_FakeRetriever):
    def __init__(self, chunks):
        super().__init__(chunks)
        self.queries = []

    def retrieve(self, query, k=5):
        self.queries.append(query)
        return super().retrieve(query, k=k)


class _FakeLLM:
    def __init__(self, tokens, rewrite=None):
        self._tokens = tokens
        self.num_ctx = 8192
        self.rewrite = rewrite
        self.rewrite_calls = 0
        self.chat_calls = 0

    def chat_stream(self, messages):
        self.chat_calls += 1
        yield from self._tokens

    def chat(self, messages):
        self.chat_calls += 1
        return "".join(self._tokens)

    def rewrite_query(self, history_text, question, is_unclear=False, timeout=None):
        self.rewrite_calls += 1
        if self.rewrite is None:
            return question
        return self.rewrite(history_text, question, is_unclear)


class _FakeTranslator:
    @staticmethod
    def normalize_query(text):
        return text


class _FakeVectorStore:
    config = None


def _system(tokens, chunks):
    return _system_with(_FakeLLM(tokens), chunks)


def _system_with(llm, chunks):
    system = UniversityQASystem(llm, _FakeVectorStore(), _FakeTranslator())
    system.retriever = _FakeRetriever(chunks)
    return system


def _collect(system, question):
    async def run():
        return [event async for event in system.query_stream(question)]

    return asyncio.run(run())


def test_registry_keeps_same_number_for_same_page():
    registry = CitationRegistry()
    chunks = [
        _chunk("a", "lecture.pdf", page=4, type="pdf"),
        _chunk("b", "lecture.pdf", page=4, type="pdf"),
        _chunk("c", "notes.docx"),
    ]

    numbers = [registry.number_for(chunk.metadata) for chunk in chunks]

    assert numbers == [1, 1, 2]
    assert registry.valid_numbers == {1, 2}
    assert registry.footer() == [
        "[1] lecture.pdf — strana 4",
        "[2] notes.docx",
    ]


def test_context_tags_match_footer_numbers():
    registry = CitationRegistry()
    chunks = [
        _chunk("a", "lecture.pdf", page=4, type="pdf"),
        _chunk("b", "other.pdf", page=9, type="pdf"),
        _chunk("c", "lecture.pdf", page=4, type="pdf"),
    ]
    for chunk in chunks:
        registry.add(chunk.metadata)

    context = UniversityQASystem._build_context(chunks, registry)

    assert "[Izvor 1 — lecture.pdf, strana 4]" in context
    assert "[Izvor 2 — other.pdf, strana 9]" in context
    assert context.count("[Izvor 1 — lecture.pdf, strana 4]") == 2
    assert registry.footer() == [
        "[1] lecture.pdf — strana 4",
        "[2] other.pdf — strana 9",
    ]


def test_filter_citations_strips_unknown_numbers():
    cleaned, cited = _filter_citations(
        "Predmet nosi 6 ESPB [Izvor 1], a vežbe su obavezne [Izvor 7]. "
        "Takođe [Izvor 1, 2].",
        valid_numbers={1, 2},
    )

    assert "[Izvor 7]" not in cleaned
    assert "[Izvor 1]" in cleaned
    assert "[Izvor 1, 2]" in cleaned
    assert cited == {1, 2}


def test_filter_citations_removes_marker_with_no_valid_number():
    cleaned, cited = _filter_citations("Nema izvora [Izvor 5].", {1, 2})
    assert cleaned == "Nema izvora."
    assert cited == set()


def test_stream_filter_withholds_partial_marker_and_strips_invalid():
    filt = _CitationStreamFilter({1, 2})

    assert filt.feed("Odgovor je [Iz") == "Odgovor je "
    assert filt.feed("vor 9]") == ""
    assert filt.feed(" i \n[Izv") == " i \n"
    assert filt.feed("or 2]") == "[Izvor 2]"
    assert filt.finish() == ""


def test_stream_filter_leaves_markdown_links_alone():
    filt = _CitationStreamFilter({1})
    assert filt.feed("[link](https://example.com)") == "[link](https://example.com)"
    assert filt.finish() == ""


def test_refusal_detection_is_diacritic_and_case_insensitive():
    assert _is_refusal(REFUSAL_PHRASE.upper())
    assert _is_refusal("nažalost, NEMAM informaciju o tome u dostupnim izvorima.")
    assert _is_refusal("Nisam pronašao relevantne informacije u bazi znanja.")
    assert _is_refusal("Na osnovu konteksta, nemam informaciju o tome.")
    assert not _is_refusal("Predmet nosi 6 ESPB [Izvor 1].")


def test_long_answer_with_stray_refusal_is_not_a_refusal():
    answer = (
        "Na OAS Informatike postoje tri smera [Izvor 1]. "
        "Studenti biraju modul pri upisu na drugu godinu [Izvor 2]. "
        "Na osnovu konteksta, nemam informaciju o tome u dostupnim izvorima."
    )

    assert not _is_refusal(answer)
    stripped = _strip_stray_refusal_sentence(answer)
    assert "nemam informaciju" not in stripped
    assert "[Izvor 1]" in stripped
    assert stripped.endswith("[Izvor 2].")


def test_answer_starting_with_na_osnovu_konteksta_with_stray_refusal_keeps_sources():
    answer = (
        "Na osnovu konteksta, mogu da vam kažem sledeće: "
        "postoje tri smera [Izvor 1]. Studenti biraju modul pri upisu "
        "na drugu godinu studija [Izvor 2]. "
        "Na osnovu konteksta, nemam informaciju o tome u dostupnim izvorima."
    )

    assert not _is_refusal(answer)
    assert _strip_stray_refusal_sentence(answer).endswith("[Izvor 2].")


def test_grounded_answer_starting_with_na_osnovu_konteksta_is_not_refusal():
    answer = "Na osnovu konteksta, predmet nosi 6 ESPB [Izvor 1]."
    assert not _is_refusal(answer)


def test_standalone_heuristic_skips_rewrite_for_explicit_questions():
    assert UniversityQASystem._looks_standalone(
        "Koliko ESPB nosi predmet Uvod u programiranje?"
    )
    assert not UniversityQASystem._looks_standalone("Koji su preduslovi za njega?")
    assert not UniversityQASystem._looks_standalone("Koliko ESPB ima?")
    assert not UniversityQASystem._looks_standalone("Objasni mi to detaljnije.")


def test_unclear_patterns_cover_demonstratives():
    assert UniversityQASystem._is_unclear_question(
        "A ko je rukovodilac tog projekta?"
    )
    assert UniversityQASystem._is_unclear_question(
        "Šta možeš da mi kažeš o njima?"
    )
    assert not UniversityQASystem._is_unclear_question(
        "Koliko ESPB nosi predmet Uvod u programiranje?"
    )


def test_rewrite_failure_uses_previous_question_for_pure_followup():
    chunks = [_chunk("Prva činjenica.", "a.pdf", page=1, type="pdf")]
    # _FakeLLM echoes the question, so the rewrite is "unchanged" and the
    # deterministic context fallback must kick in.
    system = _system(["nebitno"], chunks)
    history = [
        {"role": "user", "content": "Koliko ima smerova na OAS informatika?"},
        {"role": "assistant", "content": "Postoje dva smera."},
    ]

    standalone = system._standalone_question(
        "Šta možeš da mi kažeš o njima?", history
    )

    assert standalone == "Koliko ima smerova na OAS informatika?"


def test_rewrite_failure_combines_when_followup_has_its_own_topic():
    chunks = [_chunk("Prva činjenica.", "a.pdf", page=1, type="pdf")]
    system = _system(["nebitno"], chunks)
    history = [
        {"role": "user", "content": "Koliko ima smerova na OAS informatika?"},
        {"role": "assistant", "content": "Postoje dva smera."},
    ]

    standalone = system._standalone_question(
        "A koliko ESPB ima softversko inženjerstvo?", history
    )

    assert standalone == (
        "Koliko ima smerova na OAS informatika? "
        "A koliko ESPB ima softversko inženjerstvo?"
    )


def test_pure_referential_followup_skips_llm_rewrite():
    llm = _FakeLLM(
        ["nebitno"],
        rewrite=lambda *_: "Šta možeš da mi kažeš o smerovima?",
    )
    system = _system_with(
        llm, [_chunk("Prva činjenica.", "a.pdf", page=1, type="pdf")]
    )
    history = [
        {"role": "user", "content": "Koliko ima smerova na OAS informatici?"},
        {"role": "assistant", "content": "Postoje dva smera."},
    ]

    standalone = system._standalone_question(
        "Šta možeš da mi kažeš o njima?", history
    )

    assert standalone == "Koliko ima smerova na OAS informatici?"
    assert llm.rewrite_calls == 0


def test_followup_chain_anchors_to_last_clear_question():
    history = [
        {"role": "user", "content": "Koliko ima smerova na OAS informatici?"},
        {"role": "assistant", "content": "Postoje dva smera."},
        {"role": "user", "content": "Šta možeš da mi kažeš o njima?"},
        {"role": "assistant", "content": "Softversko inženjerstvo i informacione tehnologije."},
    ]

    assert UniversityQASystem._previous_user_question("A šta još?", history) == (
        "Koliko ima smerova na OAS informatici?"
    )


def test_rewrite_that_drifts_from_topic_is_rejected():
    llm = _FakeLLM(
        ["nebitno"],
        rewrite=lambda *_: "Koji su uslovi za upis na budžet?",
    )
    system = _system_with(
        llm, [_chunk("Prva činjenica.", "a.pdf", page=1, type="pdf")]
    )
    history = [
        {"role": "user", "content": "Koliko ESPB nosi predmet Uvod u programiranje?"},
        {
            "role": "assistant",
            "content": "Predmet Uvod u programiranje nosi 6 ESPB [Izvor 1].",
        },
    ]

    standalone = system._standalone_question("Koji su preduslovi za njega?", history)

    assert standalone == (
        "Koliko ESPB nosi predmet Uvod u programiranje? "
        "Koji su preduslovi za njega?"
    )


def test_rewrite_that_resolves_reference_and_keeps_topic_is_used():
    llm = _FakeLLM(
        ["nebitno"],
        rewrite=lambda *_: "Koji su preduslovi za predmet Uvod u programiranje?",
    )
    system = _system_with(
        llm, [_chunk("Prva činjenica.", "a.pdf", page=1, type="pdf")]
    )
    history = [
        {"role": "user", "content": "Koliko ESPB nosi predmet Uvod u programiranje?"},
        {
            "role": "assistant",
            "content": "Predmet Uvod u programiranje nosi 6 ESPB [Izvor 1].",
        },
    ]

    standalone = system._standalone_question("Koji su preduslovi za njega?", history)

    assert standalone == "Koji su preduslovi za predmet Uvod u programiranje?"


def test_dedupe_chunks_removes_repeated_text():
    chunks = [
        _chunk("isti tekst", "a.pdf", page=1),
        _chunk("isti tekst", "a.pdf", page=1),
        _chunk("drugi tekst", "b.pdf", page=2),
    ]
    unique = UniversityQASystem._dedupe_chunks(chunks)
    assert [c.metadata["source"] for c in unique] == ["a.pdf", "b.pdf"]


def test_stream_strips_invalid_citations_and_emits_structured_sources():
    chunks = [
        _chunk("Prva činjenica.", "a.pdf", page=1, type="pdf"),
        _chunk("Druga činjenica.", "b.pdf", page=2, type="pdf"),
    ]
    system = _system(
        ["Prva činjenica [Izvor ", "7]. Druga činjenica [Izvor 2]."],
        chunks,
    )

    events = _collect(system, "pitanje")
    answer = "".join(e.text for e in events if isinstance(e, TokenChunk))
    sources_events = [e for e in events if isinstance(e, SourcesChunk)]

    assert "[Izvor 7]" not in answer
    assert "[Izvor 2]" in answer
    assert len(sources_events) == 1
    numbers = [row["number"] for row in sources_events[0].sources]
    assert numbers == [2]
    assert sources_events[0].sources[0]["filename"] == "b.pdf"


def test_stream_refusal_suppresses_sources():
    chunks = [_chunk("Prva činjenica.", "a.pdf", page=1, type="pdf")]
    system = _system([REFUSAL_PHRASE], chunks)

    events = _collect(system, "nepoznato pitanje")
    answer = "".join(e.text for e in events if isinstance(e, TokenChunk))
    sources_events = [e for e in events if isinstance(e, SourcesChunk)]

    assert answer == REFUSAL_PHRASE
    assert sources_events == [SourcesChunk([])]


def test_stream_without_retrieved_chunks_refuses_immediately():
    system = _system(["nešto"], [])

    events = _collect(system, "nepoznato pitanje")

    assert events[0] == TokenChunk(REFUSAL_PHRASE)
    assert events[-1] == SourcesChunk([])


def test_unresolvable_followup_skips_retrieval_and_generation():
    llm = _FakeLLM(["nebitno"])
    system = _system_with(
        llm, [_chunk("Prva činjenica.", "a.pdf", page=1, type="pdf")]
    )
    retriever = _CountingRetriever(
        [_chunk("Prva činjenica.", "a.pdf", page=1, type="pdf")]
    )
    system.retriever = retriever

    events = _collect(system, "Šta možeš da mi kažeš o njima?")

    assert events[0] == TokenChunk(REFUSAL_PHRASE)
    assert events[-1] == SourcesChunk([])
    assert retriever.queries == []
    assert llm.chat_calls == 0
    assert llm.rewrite_calls == 0


def test_short_topical_question_without_history_retrieves_verbatim():
    system = _system(
        ["nebitno"], [_chunk("Prva činjenica.", "a.pdf", page=1, type="pdf")]
    )
    retriever = _CountingRetriever(
        [_chunk("Prva činjenica.", "a.pdf", page=1, type="pdf")]
    )
    system.retriever = retriever

    _collect(system, "Šta je ESPB?")

    assert retriever.queries == ["Šta je ESPB?"]


def test_retrieval_uses_rewritten_query_for_followup():
    llm = _FakeLLM(
        ["nebitno"],
        rewrite=lambda *_: "Koji su preduslovi za predmet Uvod u programiranje?",
    )
    system = _system_with(
        llm, [_chunk("Prva činjenica.", "a.pdf", page=1, type="pdf")]
    )
    retriever = _CountingRetriever(
        [_chunk("Prva činjenica.", "a.pdf", page=1, type="pdf")]
    )
    system.retriever = retriever
    history = [
        {"role": "user", "content": "Koliko ESPB nosi predmet Uvod u programiranje?"},
        {"role": "assistant", "content": "Predmet Uvod u programiranje nosi 6 ESPB."},
    ]

    async def run():
        return [
            event
            async for event in system.query_stream("Koji su preduslovi za njega?", history)
        ]

    asyncio.run(run())

    assert retriever.queries == [
        "Koji su preduslovi za predmet Uvod u programiranje?"
    ]
