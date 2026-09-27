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
        self.messages = None

    def chat_stream(self, messages):
        self.chat_calls += 1
        self.messages = messages
        yield from self._tokens

    def chat(self, messages):
        self.chat_calls += 1
        self.messages = messages
        return "".join(self._tokens)

    def rewrite_query(self, history_text, question, timeout=None):
        self.rewrite_calls += 1
        if self.rewrite is None:
            return question
        return self.rewrite(history_text, question)


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


def _collect(system, question, history=None):
    async def run():
        return [
            event async for event in system.query_stream(question, history)
        ]

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


def test_filter_citations_removes_all_markers_and_reports_valid_citations():
    cleaned, cited = _filter_citations(
        "Predmet nosi 6 ESPB [Izvor 1], a vežbe su obavezne [Izvor 7]. "
        "Takođe [Izvor 1, 2].",
        valid_numbers={1, 2},
    )

    assert "[Izvor" not in cleaned
    assert "[izvor" not in cleaned.lower()
    assert "Predmet nosi 6 ESPB, a vežbe su obavezne." in cleaned
    assert cited == {1, 2}


def test_filter_citations_removes_marker_with_no_valid_number():
    cleaned, cited = _filter_citations("Nema izvora [Izvor 5].", {1, 2})
    assert cleaned == "Nema izvora."
    assert cited == set()


def test_stream_filter_strips_markers_and_records_cited_numbers():
    filt = _CitationStreamFilter({1, 2})

    assert filt.feed("Odgovor je [Iz") == "Odgovor je"
    assert filt.feed("vor 9]") == ""
    assert filt.feed(" i [Izv") == " i"
    assert filt.feed("or 2]") == ""
    assert filt.feed(" tekst.") == " tekst."
    assert filt.finish() == ""
    assert filt.cited == {2}


def test_stream_filter_drops_space_left_by_removed_marker():
    filt = _CitationStreamFilter({2})

    assert filt.feed("nosi 6 ESPB ") == "nosi 6 ESPB"
    assert filt.feed("[Izvor 2]") == ""
    assert filt.feed(".") == "."
    assert filt.cited == {2}


def test_stream_filter_keeps_citations_out_of_the_text():
    filt = _CitationStreamFilter({1})

    emitted = (
        filt.feed("Prva činjenica [Izvor 1]. ")
        + filt.feed("Druga činjenica [Izvor 1].")
        + filt.finish()
    )

    assert emitted == "Prva činjenica. Druga činjenica."
    assert filt.cited == {1}


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


def test_rewrite_transcript_keeps_both_roles_and_recent_turns():
    history = [
        {"role": "user", "content": "Koliko modula ima Informatika?"},
        {"role": "assistant", "content": "Informatika ima dva modula."},
    ]

    transcript = UniversityQASystem._format_history_for_rewrite(history)

    assert "Korisnik: Koliko modula ima Informatika?" in transcript
    assert "Asistent: Informatika ima dva modula." in transcript


def test_rewrite_transcript_is_bounded():
    history = [
        {"role": "user", "content": "p" * 500},
        {"role": "assistant", "content": "a" * 5000},
    ]

    transcript = UniversityQASystem._format_history_for_rewrite(history)

    assert len(transcript) <= UniversityQASystem._REWRITE_HISTORY_CHAR_LIMIT
    assert "..." in transcript


def test_rewrite_failure_uses_previous_question_for_pure_followup():
    chunks = [_chunk("Prva činjenica.", "a.pdf", page=1, type="pdf")]
    # An empty rewrite means the operation-A call failed; the deterministic
    # context fallback must kick in for a purely referential follow-up.
    system = _system_with(
        _FakeLLM(["nebitno"], rewrite=lambda *_: ""), chunks
    )
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
    system = _system_with(
        _FakeLLM(["nebitno"], rewrite=lambda *_: ""), chunks
    )
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


def test_pure_referential_followup_is_rewritten_using_the_transcript():
    captured = {}

    def rewrite(history_text, question):
        captured["history_text"] = history_text
        captured["question"] = question
        return "Šta možeš da mi kažeš o smerovima?"

    llm = _FakeLLM(["nebitno"], rewrite=rewrite)
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

    assert standalone == "Šta možeš da mi kažeš o smerovima?"
    assert llm.rewrite_calls == 1
    assert captured["question"] == "Šta možeš da mi kažeš o njima?"
    assert "Asistent: Postoje dva smera." in captured["history_text"]
    assert "Korisnik: Koliko ima smerova na OAS informatici?" in captured["history_text"]


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


def test_rewrite_that_swaps_the_named_topic_is_rejected():
    llm = _FakeLLM(
        ["nebitno"],
        rewrite=lambda *_: "Koliko modula ima na matematici?",
    )
    system = _system_with(
        llm, [_chunk("Prva činjenica.", "a.pdf", page=1, type="pdf")]
    )
    history = [
        {"role": "user", "content": "Koliko ima modula na matematici?"},
        {
            "role": "assistant",
            "content": "Matematika ima tri modula [Izvor 1].",
        },
    ]

    standalone = system._standalone_question("A koliko ima na informatici?", history)

    assert standalone == (
        "Koliko ima modula na matematici? A koliko ima na informatici?"
    )


def test_rewrite_that_keeps_the_named_topic_and_fills_the_gap_is_used():
    llm = _FakeLLM(
        ["nebitno"],
        rewrite=lambda *_: "Koliko modula ima na informatici?",
    )
    system = _system_with(
        llm, [_chunk("Prva činjenica.", "a.pdf", page=1, type="pdf")]
    )
    history = [
        {"role": "user", "content": "Koliko ima modula na matematici?"},
        {
            "role": "assistant",
            "content": "Matematika ima tri modula [Izvor 1].",
        },
    ]

    standalone = system._standalone_question("A koliko ima na informatici?", history)

    assert standalone == "Koliko modula ima na informatici?"


def test_subject_elided_followup_is_rewritten_from_previous_answer():
    captured = {}

    def rewrite(history_text, question):
        captured["history_text"] = history_text
        return "Od kada studentska služba radi sa studentima?"

    llm = _FakeLLM(["nebitno"], rewrite=rewrite)
    system = _system_with(
        llm,
        [_chunk("Studentska služba radi od 9 do 15.", "a.pdf", page=1, type="pdf")],
    )
    history = [
        {"role": "user", "content": "Od kada radi studentska služba?"},
        {
            "role": "assistant",
            "content": "Studentska služba radi od 9 do 15 [Izvor 1].",
        },
    ]

    standalone = system._standalone_question("Od kad radi sa studentima?", history)

    assert standalone == "Od kada studentska služba radi sa studentima?"
    assert llm.rewrite_calls == 1
    assert "Studentska služba radi od 9 do 15" in captured["history_text"]


def test_semantically_standalone_question_is_used_unchanged():
    # History is present, so the semantic operation-A call runs; the model
    # decides the new question is self-contained and returns it unchanged.
    llm = _FakeLLM(["nebitno"], rewrite=lambda history_text, question: question)
    system = _system_with(
        llm,
        [_chunk("Studentska služba radi od 9 do 15.", "a.pdf", page=1, type="pdf")],
    )
    retriever = _CountingRetriever(
        [_chunk("Studentska služba radi od 9 do 15.", "a.pdf", page=1, type="pdf")]
    )
    system.retriever = retriever
    history = [
        {"role": "user", "content": "Od kada radi studentska služba?"},
        {
            "role": "assistant",
            "content": "Studentska služba radi od 9 do 15 [Izvor 1].",
        },
    ]

    _collect(system, "Od kada radi biblioteka?", history)

    assert llm.rewrite_calls == 1
    assert retriever.queries == ["Od kada radi biblioteka?"]


def test_dedupe_chunks_removes_repeated_text():
    chunks = [
        _chunk("isti tekst", "a.pdf", page=1),
        _chunk("isti tekst", "a.pdf", page=1),
        _chunk("drugi tekst", "b.pdf", page=2),
    ]
    unique = UniversityQASystem._dedupe_chunks(chunks)
    assert [c.metadata["source"] for c in unique] == ["a.pdf", "b.pdf"]


def test_stream_strips_citations_and_emits_only_cited_sources():
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

    assert "[Izvor" not in answer
    assert "Prva činjenica. Druga činjenica." in answer
    assert len(sources_events) == 1
    numbers = [row["number"] for row in sources_events[0].sources]
    assert numbers == [2]
    assert sources_events[0].sources[0]["filename"] == "b.pdf"


def test_query_strips_citations_and_returns_only_cited_sources():
    chunks = [
        _chunk("Prva činjenica.", "a.pdf", page=1, type="pdf"),
        _chunk("Druga činjenica.", "b.pdf", page=2, type="pdf"),
    ]
    system = _system(
        ["Prva činjenica [Izvor 1]. Nepostojeći izvor [Izvor 9]."], chunks
    )

    response = system.query("pitanje")

    assert "[Izvor" not in response["answer"]
    assert response["answer"] == "Prva činjenica. Nepostojeći izvor."
    assert response["sources"] == ["[1] a.pdf — strana 1"]
    numbers = [row["number"] for row in response["sources_structured"]]
    assert numbers == [1]


def test_stream_uncited_answer_attributes_only_relevant_sources():
    chunks = [
        _chunk("Prva činjenica o informatici.", "a.pdf", page=1, type="pdf"),
        _chunk("Matematika i analiza.", "matematika.pdf", page=2, type="pdf"),
    ]
    system = _system(
        ["Prva činjenica o informatici bez oznake izvora."], chunks
    )

    events = _collect(system, "pitanje o informatici")
    sources_events = [e for e in events if isinstance(e, SourcesChunk)]

    assert len(sources_events) == 1
    numbers = [row["number"] for row in sources_events[0].sources]
    assert numbers == [1]
    assert sources_events[0].sources[0]["filename"] == "a.pdf"


def test_stream_uncited_answer_without_matching_chunks_has_no_sources():
    chunks = [_chunk("Matematika i analiza.", "matematika.pdf", page=2, type="pdf")]
    system = _system(["Potpuno nepovezana informacija o vremenu."], chunks)

    events = _collect(system, "pitanje")
    sources_events = [e for e in events if isinstance(e, SourcesChunk)]

    assert sources_events == [SourcesChunk([])]


def test_stream_with_stray_refusal_keeps_cited_sources():
    chunks = [_chunk("Prva činjenica.", "a.pdf", page=1, type="pdf")]
    answer_text = (
        "Prva činjenica je tačna i potkrepljena kontekstom [Izvor 1]. "
        "Na osnovu konteksta, nemam informaciju o tome u dostupnim izvorima."
    )
    system = _system([answer_text], chunks)

    events = _collect(system, "pitanje")
    answer = "".join(e.text for e in events if isinstance(e, TokenChunk))
    sources_events = [e for e in events if isinstance(e, SourcesChunk)]

    assert "[Izvor" not in answer
    assert len(sources_events) == 1
    numbers = [row["number"] for row in sources_events[0].sources]
    assert numbers == [1]


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


def test_referential_followup_without_history_retrieves_original():
    # No previous conversation: the request is treated as standalone (no
    # rewrite call) and searched with its original wording.  The answer
    # model can refuse if the retrieved context holds nothing relevant.
    llm = _FakeLLM(["nebitno"])
    system = _system_with(
        llm, [_chunk("Prva činjenica.", "a.pdf", page=1, type="pdf")]
    )
    retriever = _CountingRetriever(
        [_chunk("Prva činjenica.", "a.pdf", page=1, type="pdf")]
    )
    system.retriever = retriever

    _collect(system, "Šta možeš da mi kažeš o njima?")

    assert retriever.queries == ["Šta možeš da mi kažeš o njima?"]
    assert llm.rewrite_calls == 0
    assert llm.chat_calls == 1


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


def test_prompt_contains_only_system_and_current_turn():
    llm = _FakeLLM(["Prva činjenica [Izvor 1]."])
    system = _system_with(
        llm, [_chunk("Prva činjenica.", "a.pdf", page=1, type="pdf")]
    )
    history = [
        {"role": "user", "content": "Koliko ESPB nosi Uvod u programiranje?"},
        {"role": "assistant", "content": "Predmet nosi 6 ESPB."},
    ]

    _collect(system, "Koliko ESPB nosi Uvod u programiranje?", history)

    assert [m["role"] for m in llm.messages] == ["system", "user"]
    user_content = llm.messages[1]["content"]
    assert "Predmet nosi 6 ESPB." not in user_content
    assert "Pitanje: Koliko ESPB nosi Uvod u programiranje?" in user_content
    assert "Prva činjenica." in user_content


def test_prompt_uses_resolved_question_for_followup():
    llm = _FakeLLM(
        ["Prva činjenica [Izvor 1]."],
        rewrite=lambda *_: "Koji su preduslovi za predmet Uvod u programiranje?",
    )
    system = _system_with(
        llm, [_chunk("Prva činjenica.", "a.pdf", page=1, type="pdf")]
    )
    history = [
        {"role": "user", "content": "Koliko ESPB nosi predmet Uvod u programiranje?"},
        {"role": "assistant", "content": "Predmet Uvod u programiranje nosi 6 ESPB."},
    ]

    _collect(system, "Koji su preduslovi za njega?", history)

    assert [m["role"] for m in llm.messages] == ["system", "user"]
    user_content = llm.messages[1]["content"]
    assert (
        "Pitanje: Koji su preduslovi za predmet Uvod u programiranje?"
        in user_content
    )
    assert "Koji su preduslovi za njega?" not in user_content
    assert "Predmet Uvod u programiranje nosi 6 ESPB." not in user_content


# ── required end-to-end scenarios ───────────────────────────────────── #


def _scenario_system(llm, retriever_chunks):
    system = _system_with(llm, retriever_chunks)
    retriever = _CountingRetriever(retriever_chunks)
    system.retriever = retriever
    return system, retriever


def test_scenario_1_standalone_question_needs_no_rewrite():
    chunk = _chunk(
        "Informatika ima dva modula: Softversko inženjerstvo i "
        "Informacioni sistemi.",
        "informatika.pdf",
        page=1,
        type="pdf",
    )
    llm = _FakeLLM(["Informatika ima dva modula [Izvor 1]."])
    system, retriever = _scenario_system(llm, [chunk])

    _collect(system, "Koliko modula ima Informatika?")

    # No history → no operation-A call, search uses the question as-is and
    # the answer prompt carries only the question plus the retrieved chunk.
    assert llm.rewrite_calls == 0
    assert retriever.queries == ["Koliko modula ima Informatika?"]
    assert [m["role"] for m in llm.messages] == ["system", "user"]
    user_content = llm.messages[1]["content"]
    assert "Pitanje: Koliko modula ima Informatika?" in user_content
    assert "Informatika ima dva modula: Softversko" in user_content


def test_scenario_2_followup_is_rewritten_before_search_and_answer():
    rewritten = (
        "Koja su dva modula studijskog programa Informatika i koje su "
        "njihove karakteristike?"
    )
    chunk = _chunk(
        "Moduli su Softversko inženjerstvo i Informacioni sistemi.",
        "informatika.pdf",
        page=1,
        type="pdf",
    )
    llm = _FakeLLM(
        ["Moduli su Softversko inženjerstvo i Informacioni sistemi [Izvor 1]."],
        rewrite=lambda *_: rewritten,
    )
    system, retriever = _scenario_system(llm, [chunk])
    history = [
        {"role": "user", "content": "Koliko modula ima Informatika?"},
        {
            "role": "assistant",
            "content": "Informatika ima tačno dva modula.",
        },
    ]

    _collect(system, "Možeš li mi reći nešto više o njima?", history)

    assert llm.rewrite_calls == 1
    assert retriever.queries == [rewritten]
    assert [m["role"] for m in llm.messages] == ["system", "user"]
    user_content = llm.messages[1]["content"]
    assert f"Pitanje: {rewritten}" in user_content
    # The conversation history never reaches the answer model.
    assert "Informatika ima tačno dva modula." not in user_content
    assert "Možeš li mi reći nešto više o njima?" not in user_content


def test_scenario_3_pronoun_reference_is_resolved_from_history():
    rewritten = "Gde se nalazi kabinet profesorke Višnje?"
    chunk = _chunk(
        "Kabinet profesorice Višnje nalazi se na drugom spratu.",
        "profesori.pdf",
        page=2,
        type="pdf",
    )
    llm = _FakeLLM(
        ["Kabinet se nalazi na drugom spratu [Izvor 1]."],
        rewrite=lambda *_: rewritten,
    )
    system, retriever = _scenario_system(llm, [chunk])
    history = [
        {"role": "user", "content": "Ko je profesor Višnja?"},
        {
            "role": "assistant",
            "content": "Višnja je profesor na Institutu za matematiku i informatiku.",
        },
    ]

    _collect(system, "Gde mogu da pronađem njen kabinet?", history)

    assert llm.rewrite_calls == 1
    assert retriever.queries == [rewritten]
    user_content = llm.messages[1]["content"]
    assert f"Pitanje: {rewritten}" in user_content
    assert "Višnja je profesor na Institutu" not in user_content


def test_scenario_4_unrelated_question_is_not_influenced_by_history():
    question = "Koji su uslovi za upis na master studije?"
    chunk = _chunk(
        "Uslovi za upis na master studije su završene osnovne studije.",
        "upis.pdf",
        page=3,
        type="pdf",
    )
    # The semantic operation-A call returns the question unchanged because
    # it is self-contained; the professor answer must not leak into the
    # retrieval query or the final answer prompt.
    llm = _FakeLLM(
        ["Za upis na master studije potrebne su završene osnovne studije [Izvor 1]."],
        rewrite=lambda history_text, q: q,
    )
    system, retriever = _scenario_system(llm, [chunk])
    history = [
        {"role": "user", "content": "Ko je profesor Višnja?"},
        {
            "role": "assistant",
            "content": "Višnja je profesor na Institutu za matematiku i informatiku.",
        },
    ]

    _collect(system, question, history)

    assert llm.rewrite_calls == 1
    assert retriever.queries == [question]
    user_content = llm.messages[1]["content"]
    assert f"Pitanje: {question}" in user_content
    assert "Višnja" not in user_content
    assert "profesor na Institutu" not in user_content


def test_scenario_5_followup_resolves_reference_across_multiple_turns():
    rewritten = "Gde se nalazi kabinet profesorke Višnje?"
    captured = {}

    def rewrite(history_text, question):
        captured["history_text"] = history_text
        return rewritten

    chunk = _chunk(
        "Kabinet profesorice Višnje nalazi se na drugom spratu.",
        "profesori.pdf",
        page=2,
        type="pdf",
    )
    llm = _FakeLLM(["Kabinet se nalazi na drugom spratu [Izvor 1]."], rewrite=rewrite)
    system, retriever = _scenario_system(llm, [chunk])
    history = [
        {"role": "user", "content": "Koji su uslovi za upis?"},
        {"role": "assistant", "content": "Za upis su potrebni dokumenti."},
        {"role": "user", "content": "Ko je profesor Višnja?"},
        {
            "role": "assistant",
            "content": "Višnja je profesor na Institutu za matematiku i informatiku.",
        },
    ]

    _collect(system, "A gde je njen kabinet?", history)

    # The transcript for operation A spans several turns, so the reference
    # can be resolved even though the entity was named two turns back.
    assert "Koji su uslovi za upis?" in captured["history_text"]
    assert "Ko je profesor Višnja?" in captured["history_text"]
    assert "Višnja je profesor" in captured["history_text"]
    assert retriever.queries == [rewritten]
