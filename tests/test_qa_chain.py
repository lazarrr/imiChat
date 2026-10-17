"""
Unit tests for UniversityQASystem's question handling.

The answering LLM must only see the system prompt, the retrieved chunks and
one question — never the conversation history.  History is used only to
rewrite unclear follow-ups ("njih", "o njima", ...).

No Ollama or ChromaDB needed: the LLM manager and vector store are mocks.

    python -m unittest tests.test_qa_chain -v     (or: pytest tests/test_qa_chain.py)
"""

import asyncio
import importlib.util
import re
import threading
import unittest
from unittest.mock import MagicMock

from src.chains.qa_chain import UniversityQASystem

PREVIOUS_QUESTION = "Koji su moduli na osnovnim studijama informatike?"
PREVIOUS_ANSWER = (
    "Na osnovnim studijama informatike postoje dva modula: "
    "Racunarske nauke i Softversko inzenjerstvo."
)
HISTORY = [
    {"role": "user", "content": PREVIOUS_QUESTION},
    {"role": "assistant", "content": PREVIOUS_ANSWER},
]
REWRITTEN = (
    "Daj mi vise informacija o modulima Racunarske nauke i "
    "Softversko inzenjerstvo na osnovnim studijama informatike"
)
CHUNK = "Modul Softversko inzenjerstvo obuhvata razvoj i testiranje softvera."

HAS_TRANSLATOR_DEPS = all(
    importlib.util.find_spec(name) for name in ("cyrtranslit", "langdetect")
)


class IdentityTranslator:
    def normalize_query(self, text: str) -> str:
        return text


# Metadata as the vector store actually holds it, extra keys included.
CHUNK_META = {
    "source": "informatika.pdf", "filename": "informatika.pdf",
    "type": "pdf", "page": 3, "url": "",
    "llm_processed": False, "llm_original_chars": 830,
}

EXCLUDED_CHUNK = {
    "chunk_id": "abc123", "reason": "excluded_threshold", "distance": 0.44,
    "metadata": {"filename": "stari_plan.pdf", "status": "legacy"},
    "text": "Stari plan studija.", "threshold": 0.3,
}


def make_system(rewrite=REWRITTEN, translator=None):
    llm = MagicMock()
    llm.rewrite_query.return_value = rewrite
    # None until rewrite_query() runs; tests that assert on the logged prompt
    # set it explicitly, like the real LLMManager does.
    llm.last_rewrite = None
    llm.chat.return_value = "Odgovor."
    llm.chat_stream.side_effect = lambda messages: iter(["Odg", "ovor."])

    store = MagicMock()
    store.similarity_search.return_value = {
        "documents": [[CHUNK]],
        "metadatas": [[dict(CHUNK_META)]],
        "distances": [[0.21]],
        "excluded": [dict(EXCLUDED_CHUNK)],
        "excluded_total": 1,
    }
    qa = UniversityQASystem(llm, store, translator or IdentityTranslator())
    return qa, llm, store


async def drain(stream):
    """All tokens of an async token stream."""
    return [token async for token in stream]


def collect(stream):
    """Run an async token stream to completion and return its tokens."""
    return asyncio.run(drain(stream))


def on_event_loop() -> bool:
    """True when called from the thread that runs the asyncio event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class QuestionHandlingTests(unittest.TestCase):
    def assert_only_question_and_chunks(self, messages, expected_question):
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        user_content = messages[1]["content"]
        self.assertIn(CHUNK, user_content)
        self.assertTrue(user_content.endswith(f"Pitanje: {expected_question}"))
        for message in messages:
            self.assertNotIn(PREVIOUS_QUESTION, message["content"])
            self.assertNotIn(PREVIOUS_ANSWER, message["content"])

    def test_clear_follow_up_is_not_rewritten_and_history_not_sent(self):
        qa, llm, store = make_system()
        question = "Koliko ESPB ima Uvod u programiranje?"

        result = qa.query(question, HISTORY)

        llm.rewrite_query.assert_not_called()
        # k=None → the vector store uses TOP_K_RESULTS; diagnostics are only
        # requested when a trace is being filled in.
        store.similarity_search.assert_called_once_with(
            question, k=None, diagnostics=False)
        self.assert_only_question_and_chunks(llm.chat.call_args.args[0], question)
        self.assertEqual(result["answer"], "Odgovor.")

    def test_unclear_follow_up_is_rewritten_and_only_rewrite_is_sent(self):
        qa, llm, store = make_system()
        question = "Daj mi više informacija o njima"

        qa.query(question, HISTORY)

        llm.rewrite_query.assert_called_once()
        history_text, rewritten_from = llm.rewrite_query.call_args.args
        # The rewrite gets the whole last turn: the question (topic) and
        # the answer (what "njima" refers to).
        self.assertIn(PREVIOUS_QUESTION, history_text)
        self.assertIn(PREVIOUS_ANSWER, history_text)
        self.assertEqual(rewritten_from, question)
        self.assertEqual(store.similarity_search.call_args_list[0].args[0], REWRITTEN)
        messages = llm.chat.call_args.args[0]
        self.assert_only_question_and_chunks(messages, REWRITTEN)
        self.assertNotIn(question, messages[1]["content"])

    def test_unclear_question_without_history_is_not_rewritten(self):
        qa, llm, _ = make_system()
        question = "Daj mi više informacija o njima"

        qa.query(question, [])

        llm.rewrite_query.assert_not_called()
        self.assert_only_question_and_chunks(llm.chat.call_args.args[0], question)

    def test_history_for_rewrite_keeps_the_previous_question_whole(self):
        # The previous question is what names the topic, so it survives
        # longer than the 400-char cap older answers get.
        long_question = "a" * 700
        history = [
            {"role": "user", "content": long_question},
            {"role": "assistant", "content": PREVIOUS_ANSWER},
            {"role": "user", "content": "A koliko traje?"},
        ]

        text = UniversityQASystem._format_history_for_rewrite(history)

        self.assertIn(f"Korisnik: {long_question[:500]}...", text)
        self.assertNotIn(long_question, text)
        self.assertIn(f"Asistent: {PREVIOUS_ANSWER}", text)
        self.assertTrue(text.endswith("Korisnik: A koliko traje?"))

    def test_history_for_rewrite_keeps_a_short_previous_question_verbatim(self):
        text = UniversityQASystem._format_history_for_rewrite(HISTORY)

        self.assertIn(f"Korisnik: {PREVIOUS_QUESTION}", text)
        self.assertNotIn("...", text)

    def test_stream_sends_only_rewritten_question_and_chunks(self):
        qa, llm, _ = make_system()

        tokens = collect(qa.query_stream("Daj mi više informacija o njima", HISTORY))

        self.assert_only_question_and_chunks(llm.chat_stream.call_args.args[0], REWRITTEN)
        self.assertEqual(tokens[:2], ["Odg", "ovor."])
        self.assertIn("Izvori:", tokens[-1])

    def test_short_question_accepts_long_rewrite(self):
        qa, llm, _ = make_system()

        qa.query("A o njima?", HISTORY)

        self.assert_only_question_and_chunks(llm.chat.call_args.args[0], REWRITTEN)

    def test_rewrite_that_looks_like_an_answer_falls_back_to_original(self):
        qa, llm, _ = make_system(rewrite="Odgovor koji je predug. " * 20)

        qa.query("A o njima?", HISTORY)

        self.assert_only_question_and_chunks(llm.chat.call_args.args[0], "A o njima?")

    def test_unclear_question_falls_back_to_previous_question_then_answer(self):
        qa, llm, store = make_system()
        hit = store.similarity_search.return_value
        empty = {"documents": [[]], "metadatas": [[]]}
        store.similarity_search.side_effect = [empty, empty, hit]

        qa.query("Daj mi više informacija o njima", HISTORY)

        queries = [c.args[0] for c in store.similarity_search.call_args_list]
        self.assertEqual(len(queries), 3)
        self.assertEqual(queries[:2], [REWRITTEN, "Daj mi više informacija o njima"])
        self.assertEqual(queries[2], f"{PREVIOUS_QUESTION} Daj mi više informacija o njima")
        # Retrieval may use the history, the answering LLM still does not.
        self.assert_only_question_and_chunks(llm.chat.call_args.args[0], REWRITTEN)

    def test_last_answer_fallback_is_used_when_the_previous_question_fails(self):
        qa, llm, store = make_system()
        hit = store.similarity_search.return_value
        empty = {"documents": [[]], "metadatas": [[]]}
        store.similarity_search.side_effect = [empty, empty, empty, hit]

        qa.query("Daj mi više informacija o njima", HISTORY)

        queries = [c.args[0] for c in store.similarity_search.call_args_list]
        self.assertEqual(len(queries), 4)
        self.assertEqual(queries[2], f"{PREVIOUS_QUESTION} Daj mi više informacija o njima")
        self.assertTrue(queries[3].startswith(PREVIOUS_ANSWER))
        self.assert_only_question_and_chunks(llm.chat.call_args.args[0], REWRITTEN)

    def test_clear_question_with_no_results_does_not_use_history(self):
        qa, llm, store = make_system()
        store.similarity_search.return_value = {"documents": [[]], "metadatas": [[]]}

        result = qa.query("Koliko ESPB ima Uvod u programiranje?", HISTORY)

        store.similarity_search.assert_called_once()
        llm.chat.assert_not_called()
        self.assertEqual(result["sources"], [])


class EventLoopTests(unittest.TestCase):
    """The rewrite, retrieval and the answer stream are blocking calls.  On
    the event loop they would make the API answer one request at a time."""

    def test_blocking_steps_run_off_the_event_loop(self):
        qa, llm, store = make_system()
        hit = store.similarity_search.return_value
        on_loop = {}

        def rewrite(history_text, question):
            on_loop["rewrite"] = on_event_loop()
            return REWRITTEN

        def search(query, k=None, diagnostics=False):
            on_loop["search"] = on_event_loop()
            return hit

        def stream(messages):
            on_loop["answer"] = on_event_loop()
            yield "Odgovor."

        llm.rewrite_query.side_effect = rewrite
        store.similarity_search.side_effect = search
        llm.chat_stream.side_effect = stream

        collect(qa.query_stream("Daj mi više informacija o njima", HISTORY))

        self.assertEqual(on_loop, {"rewrite": False, "search": False, "answer": False})

    def test_two_answers_stream_at_the_same_time(self):
        # Each answer waits until the other one is streaming too.  If one
        # stream held the event loop, the other request could not start
        # and the barrier would time out.
        both_streaming = threading.Barrier(2, timeout=5)

        def stream(messages):
            both_streaming.wait()
            yield "Odgovor."

        qa, llm, _ = make_system()
        llm.chat_stream.side_effect = stream

        async def two_requests():
            return await asyncio.gather(
                drain(qa.query_stream("Koliko ESPB ima Uvod u programiranje?", [])),
                drain(qa.query_stream("Kada je upis na master studije?", [])),
            )

        for tokens in asyncio.run(two_requests()):
            self.assertEqual(tokens[0], "Odgovor.")


class TraceTests(unittest.TestCase):
    """The pipeline trace is what the request log is built from, so the
    chain must fill in the rewrite, every retrieval attempt and the prompt."""

    def test_trace_records_rewrite_retrieval_and_prompt(self):
        qa, llm, store = make_system()
        llm.last_rewrite = {
            "prompt": "PROMPT SA ISTORIJOM",
            "raw_response": REWRITTEN,
            "history_text": f"Korisnik: {PREVIOUS_QUESTION}\nAsistent: {PREVIOUS_ANSWER}",
            "latest_question": "Daj mi više informacija o njima",
        }
        trace = {}

        result = qa.query("Daj mi više informacija o njima", HISTORY, trace=trace)

        self.assertEqual(trace["rewrite"]["question_in"], "Daj mi više informacija o njima")
        self.assertEqual(trace["rewrite"]["question_out"], REWRITTEN)
        self.assertEqual(trace["rewrite"]["prompt"], "PROMPT SA ISTORIJOM")
        self.assertEqual(trace["rewrite"]["result"], "accepted")
        self.assertTrue(trace["rewrite"]["changed"])
        self.assertIn(PREVIOUS_ANSWER, trace["rewrite"]["history_text"])

        self.assertEqual(trace["search_question"], REWRITTEN)
        self.assertTrue(trace["unclear"])
        self.assertEqual(trace["retrieval"]["winning_strategy"], "search_question")
        self.assertEqual(trace["retrieval"]["attempts"], [{
            "rank": 1, "strategy": "search_question", "query": REWRITTEN,
            "query_chars": len(REWRITTEN), "hits": 1, "top_distance": 0.21,
            "excluded": 1,
        }])

        self.assertEqual(len(trace["chunks"]), 1)
        self.assertEqual(trace["chunks"][0]["filename"], "informatika.pdf")
        self.assertEqual(trace["chunks"][0]["text"], CHUNK)
        self.assertEqual(trace["chunks"][0]["rank"], 1)
        # Every metadata field the vector store holds is kept, including the
        # ones this project never looks at.
        self.assertEqual(trace["chunks"][0]["metadata"]["llm_original_chars"], 830)
        self.assertEqual(trace["chunks"][0]["metadata"], CHUNK_META)

        # The prompt recorded is exactly what the model got.
        self.assertEqual(trace["prompt"]["messages"], llm.chat.call_args.args[0])
        self.assertIn(CHUNK, trace["prompt"]["messages"][1]["content"])
        # Both entry points record the citation lines the answer carried.
        self.assertEqual(trace["sources"], ["[1] informatika.pdf — strana 3"])
        self.assertIn("rewrite", trace["timings_ms"])
        self.assertIn("retrieval", trace["timings_ms"])
        self.assertIn("answer", trace["timings_ms"])
        self.assertEqual(result["answer"], "Odgovor.")

    def test_trace_records_a_clear_question_as_skipped(self):
        qa, _, _ = make_system()
        trace = {}

        qa.query("Koliko ESPB ima Uvod u programiranje?", HISTORY, trace=trace)

        self.assertEqual(trace["rewrite"]["result"], "skipped_clear_question")
        self.assertFalse(trace["rewrite"]["changed"])
        self.assertIsNone(trace["rewrite"]["prompt"])
        self.assertIsNone(trace["rewrite"]["matched_pattern"])
        self.assertFalse(trace["unclear"])
        self.assertEqual(trace["search_question"], "Koliko ESPB ima Uvod u programiranje?")

    def test_trace_records_rejected_rewrite_with_the_guard(self):
        qa, _, _ = make_system(rewrite="")
        trace = {}

        qa.query("A o njima?", HISTORY, trace=trace)

        self.assertEqual(trace["rewrite"]["result"], "empty")
        self.assertEqual(trace["rewrite"]["question_out"], "A o njima?")
        self.assertFalse(trace["rewrite"]["changed"])
        self.assertEqual(trace["rewrite"]["rejected"], "")

    def test_trace_records_every_failed_retrieval_attempt(self):
        qa, llm, store = make_system()
        llm.last_rewrite = {"prompt": "P", "raw_response": REWRITTEN,
                            "history_text": "h", "latest_question": "q"}
        empty = {"documents": [[]], "metadatas": [[]]}
        store.similarity_search.side_effect = [
            empty, empty, empty, store.similarity_search.return_value,
        ]
        trace = {}

        qa.query("Daj mi više informacija o njima", HISTORY, trace=trace)

        attempts = trace["retrieval"]["attempts"]
        self.assertEqual([a["strategy"] for a in attempts], [
            "search_question", "original_question",
            "previous_question_plus_current", "previous_answer_plus_current",
        ])
        self.assertEqual([a["hits"] for a in attempts], [0, 0, 0, 1])
        self.assertEqual([a["rank"] for a in attempts], [1, 2, 3, 4])
        self.assertEqual(trace["retrieval"]["winning_strategy"],
                         "previous_answer_plus_current")

    def test_trace_keeps_the_chunks_that_were_not_used(self):
        # The store reports what it dropped (threshold, legacy, reranker…);
        # the trace must carry it, tagged with the attempt that produced it.
        qa, llm, store = make_system()

        trace = {}
        qa.query("Daj mi više informacija o njima", HISTORY, trace=trace)

        excluded = trace["retrieval"]["excluded"]
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["reason"], "excluded_threshold")
        self.assertEqual(excluded[0]["metadata"]["status"], "legacy")
        self.assertEqual(excluded[0]["attempt_rank"], 1)
        self.assertEqual(excluded[0]["attempt_strategy"], "search_question")
        # The winning attempt owns the exclusions that matter for the answer.
        self.assertTrue(excluded[0]["attempt_kept"])
        self.assertEqual(trace["chunks_excluded"][0]["chunk_id"], "abc123")
        self.assertEqual(trace["chunks_excluded_total"], 1)

    def test_diagnostics_are_only_requested_when_tracing(self):
        qa, _, store = make_system()
        trace = {}

        qa.query("Daj mi više informacija o njima", HISTORY, trace=trace)

        self.assertTrue(store.similarity_search.call_args.kwargs["diagnostics"])

    def test_exclusions_from_a_losing_attempt_are_marked_as_such(self):
        qa, _, store = make_system()
        empty = {"documents": [[]], "metadatas": [[]], "distances": [[]],
                 "excluded": [{**EXCLUDED_CHUNK, "reason": "excluded_legacy_version"}],
                 "excluded_total": 1}
        store.similarity_search.side_effect = [empty, store.similarity_search.return_value]
        trace = {}

        qa.query("Daj mi više informacija o njima", HISTORY, trace=trace)

        excluded = trace["retrieval"]["excluded"]
        self.assertEqual(len(excluded), 2)
        # Attempt 1 found nothing; its exclusions belong to a dead query.
        self.assertEqual(excluded[0]["attempt_rank"], 1)
        self.assertFalse(excluded[0]["attempt_kept"])
        # Attempt 2 answered the question.
        self.assertEqual(excluded[1]["attempt_rank"], 2)
        self.assertTrue(excluded[1]["attempt_kept"])
        self.assertEqual(trace["retrieval"]["excluded_total"], 2)

    def test_a_store_without_diagnostics_still_works(self):
        # An older/other vector store returning only documents+metadatas.
        qa, _, store = make_system()
        store.similarity_search.return_value = {
            "documents": [[CHUNK]], "metadatas": [[dict(CHUNK_META)]],
        }
        trace = {}

        result = qa.query("Daj mi više informacija o njima", HISTORY, trace=trace)

        self.assertEqual(result["answer"], "Odgovor.")
        self.assertEqual(trace["chunks"][0]["distance"], None)
        self.assertNotIn("excluded", trace.get("retrieval", {}))

    def test_stream_trace_is_complete_after_the_stream_ends(self):
        qa, llm, _ = make_system()
        trace = {}

        collect(qa.query_stream("Daj mi više informacija o njima", HISTORY, trace))

        self.assertEqual(trace["search_question"], REWRITTEN)
        self.assertEqual(trace["rewrite"]["question_out"], REWRITTEN)
        self.assertEqual(trace["sources"], ["[1] informatika.pdf — strana 3"])
        # Written by query_stream's finally block, once the last token is out.
        self.assertIn("total", trace["timings_ms"])

    def test_stream_trace_survives_a_client_that_stops_reading(self):
        qa, llm, _ = make_system()
        trace = {}

        async def stop_after_first_token():
            stream = qa.query_stream("Daj mi više informacija o njima", HISTORY, trace)
            first = await stream.__anext__()
            await stream.aclose()  # client hung up
            return first

        self.assertEqual(asyncio.run(stop_after_first_token()), "Odg")
        # Whatever did run is still logged, so the turn is diagnosable.
        self.assertEqual(trace["search_question"], REWRITTEN)
        self.assertEqual(trace["rewrite"]["result"], "accepted")
        self.assertIn("total", trace["timings_ms"])

    def test_trace_is_optional_and_does_not_change_the_answer(self):
        qa, _, _ = make_system()

        result = qa.query("Daj mi više informacija o njima", HISTORY)

        self.assertEqual(result["answer"], "Odgovor.")


class SourceNumberingTests(unittest.TestCase):
    METAS = [
        {"source": "a.pdf#page=1", "filename": "a.pdf", "type": "pdf", "page": 1},
        {"source": "a.pdf#page=1", "filename": "a.pdf", "type": "pdf", "page": 1},
        {"source": "https://imi.pmf.kg.ac.rs/upis#upis", "filename": "Upis", "type": "web",
         "page": 0, "url": "https://imi.pmf.kg.ac.rs/upis"},
        {"source": "a.pdf#page=2", "filename": "a.pdf", "type": "pdf", "page": 2},
    ]

    def test_context_tags_use_the_source_list_numbers(self):
        qa, _, _ = make_system()

        context = qa._build_context(["c1", "c2", "c3", "c4"], self.METAS)
        sources = qa._format_sources(self.METAS)

        # Two chunks from the same page share [Izvor 1]; [Izvor 3] is the
        # third line of the Izvori block, not the fourth chunk.
        self.assertEqual(re.findall(r"\[Izvor (\d+) — ([^\]]+)\]", context), [
            ("1", "a.pdf, strana 1"),
            ("1", "a.pdf, strana 1"),
            ("2", "Upis (https://imi.pmf.kg.ac.rs/upis)"),
            ("3", "a.pdf, strana 2"),
        ])
        self.assertEqual(sources, [
            "[1] a.pdf — strana 1",
            "[2] Upis — https://imi.pmf.kg.ac.rs/upis",
            "[3] a.pdf — strana 2",
        ])


class NoInfoAnswerTests(unittest.TestCase):
    NO_INFO = [
        "Nemam informaciju o tome.",
        "Nažalost, u priloženom kontekstu nema informacija o ceni školarine.",
        "Nisam pronašao podatke o tome u dostupnim izvorima.",
        "Kontekst ne sadrži odgovor na ovo pitanje.",
        "Ne mogu da odgovorim na ovo pitanje na osnovu dostupnih informacija.",
        "I don't have information about that.",
    ]
    REAL_ANSWERS = [
        "Predmet Osnovi programiranja nosi 6 ESPB [Izvor 1].",
        "Predmet nema preduslova.",
        # partly answered — it cites a source, so the sources stay
        "Nemam informaciju o ceni, ali upis počinje 1. jula [Izvor 1].",
        # long answer that mentions a gap in passing
        "Upis počinje 1. jula. " * 20 + "Za ostale rokove nema informacija.",
    ]

    def test_no_info_answers_are_detected(self):
        for answer in self.NO_INFO:
            with self.subTest(answer=answer):
                self.assertTrue(UniversityQASystem._is_no_info_answer(answer))

    def test_real_answers_are_not_flagged(self):
        for answer in self.REAL_ANSWERS:
            with self.subTest(answer=answer[:60]):
                self.assertFalse(UniversityQASystem._is_no_info_answer(answer))

    def test_no_info_answer_gets_no_sources(self):
        qa, llm, _ = make_system()
        llm.chat.return_value = "Nemam informaciju o tome."

        result = qa.query("Koliko košta školarina?", [])

        self.assertEqual(result["sources"], [])

    def test_no_info_stream_has_no_sources_block(self):
        qa, llm, _ = make_system()
        llm.chat_stream.side_effect = lambda messages: iter(["Nemam informaciju", " o tome."])

        tokens = collect(qa.query_stream("Koliko košta školarina?", []))

        self.assertEqual("".join(tokens), "Nemam informaciju o tome.")

    def test_partial_answer_with_citation_keeps_sources(self):
        qa, llm, _ = make_system()
        llm.chat.return_value = "Nemam informaciju o ceni, ali upis počinje 1. jula [Izvor 1]."

        result = qa.query("Koliko košta školarina i kada je upis?", [])

        self.assertEqual(result["sources"], ["[1] informatika.pdf — strana 3"])


class UnclearDetectionTests(unittest.TestCase):
    UNCLEAR = [
        "Daj mi više informacija o njima",
        "Koji su preduslovi za njega?",
        "Šta je sa njima?",
        "daj mi vise o tome",
        "A koliko traje?",
        "Nabroj ih",
        "Tell me more about them",
        # Short follow-ups named by the rewrite prompt.
        "A oni su?",
        "A koji su?",
        "A gde?",
        "Može malo više?",
        "A šta je sa tim?",
    ]
    CLEAR = [
        "Koliko ESPB ima Uvod u programiranje?",
        "Kada je upis na master studije?",
        "Ko je dekan fakulteta?",
        "Koji su moduli na osnovnim studijama informatike?",
    ]

    def test_unclear_questions_are_detected(self):
        for question in self.UNCLEAR:
            with self.subTest(question=question):
                self.assertTrue(UniversityQASystem._is_unclear_question(question))

    def test_clear_questions_are_not_flagged(self):
        for question in self.CLEAR:
            with self.subTest(question=question):
                self.assertFalse(UniversityQASystem._is_unclear_question(question))

    @unittest.skipUnless(HAS_TRANSLATOR_DEPS, "cyrtranslit / langdetect not installed")
    def test_cyrillic_unclear_question_is_rewritten(self):
        from src.processing.translator import ScriptTranslator

        qa, llm, _ = make_system(translator=ScriptTranslator())

        qa.query("Дај ми више информација о њима", HISTORY)

        llm.rewrite_query.assert_called_once()


if __name__ == "__main__":
    unittest.main()
