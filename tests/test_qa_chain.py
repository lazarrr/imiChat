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


def make_system(rewrite=REWRITTEN, translator=None):
    llm = MagicMock()
    llm.rewrite_query.return_value = rewrite
    llm.chat.return_value = "Odgovor."
    llm.chat_stream.side_effect = lambda messages: iter(["Odg", "ovor."])

    store = MagicMock()
    store.similarity_search.return_value = {
        "documents": [[CHUNK]],
        "metadatas": [[{"source": "informatika.pdf", "filename": "informatika.pdf",
                        "type": "pdf", "page": 3}]],
    }
    qa = UniversityQASystem(llm, store, translator or IdentityTranslator())
    return qa, llm, store


def collect(stream):
    """Run an async token stream to completion and return its tokens."""
    async def run():
        return [token async for token in stream]
    return asyncio.run(run())


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
        # k=None → the vector store uses TOP_K_RESULTS
        store.similarity_search.assert_called_once_with(question, k=None)
        self.assert_only_question_and_chunks(llm.chat.call_args.args[0], question)
        self.assertEqual(result["answer"], "Odgovor.")

    def test_unclear_follow_up_is_rewritten_and_only_rewrite_is_sent(self):
        qa, llm, store = make_system()
        question = "Daj mi više informacija o njima"

        qa.query(question, HISTORY)

        llm.rewrite_query.assert_called_once()
        history_text, rewritten_from = llm.rewrite_query.call_args.args
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

    def test_unclear_question_with_no_results_falls_back_to_last_answer(self):
        qa, llm, store = make_system()
        hit = store.similarity_search.return_value
        empty = {"documents": [[]], "metadatas": [[]]}
        store.similarity_search.side_effect = [empty, empty, hit]

        qa.query("Daj mi više informacija o njima", HISTORY)

        queries = [c.args[0] for c in store.similarity_search.call_args_list]
        self.assertEqual(queries[:2], [REWRITTEN, "Daj mi više informacija o njima"])
        self.assertTrue(queries[2].startswith(PREVIOUS_ANSWER))
        # Retrieval may use the history, the answering LLM still does not.
        self.assert_only_question_and_chunks(llm.chat.call_args.args[0], REWRITTEN)

    def test_clear_question_with_no_results_does_not_use_history(self):
        qa, llm, store = make_system()
        store.similarity_search.return_value = {"documents": [[]], "metadatas": [[]]}

        result = qa.query("Koliko ESPB ima Uvod u programiranje?", HISTORY)

        store.similarity_search.assert_called_once()
        llm.chat.assert_not_called()
        self.assertEqual(result["sources"], [])


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
