"""
Tests for the retrieval evaluation behind `python run.py eval`
(src/utils/retrieval_eval.py).

    python -m unittest tests.test_retrieval_eval -v
"""

import importlib.util
import re
import tempfile
import unittest
from pathlib import Path

from src.utils.retrieval_eval import evaluate, find_rank, load_cases, print_comparison

CASES = [
    {"question": "Koliko modula ima informatika?", "expect": {"text": ["dva izborna modula"]}},
    {"question": "Ko je dekan?", "expect": {"text": ["dekan fakulteta"]}},
]


class FakeStore:
    """Answers from results(query, flags); records the flags of every search."""

    def __init__(self, results):
        self.results = results
        self.use_synonyms = self.use_hybrid = self.use_reranker = False
        self.searched_flags = []

    def similarity_search(self, query, k=None):
        flags = (self.use_synonyms, self.use_hybrid, self.use_reranker)
        self.searched_flags.append(flags)
        docs, metas = self.results(query, flags)
        return {"documents": [docs[:k]], "metadatas": [metas[:k]]}


def hybrid_helps_reranker_breaks(query, flags):
    _, use_hybrid, use_reranker = flags
    if use_reranker:
        raise RuntimeError("model download failed")
    if query.startswith("Koliko"):
        docs = ["Nešto drugo.", "Opšte informacije.", "Informatika ima DVA izborna modula."]
        return (docs[::-1] if use_hybrid else docs), [{}] * 3
    return ["Raspored ispita."], [{}]  # the second question is never found


class FindRankTests(unittest.TestCase):
    def test_text_match_ignores_case_diacritics_and_punctuation(self):
        rank = find_rank(
            {"text": ["Predmet: Računarske mreže"]},
            ["Uvod.", "PREDMET:  racunarske MREZE, 6 ESPB"],
            [{}, {}],
        )
        self.assertEqual(rank, 2)

    def test_breadcrumb_and_url_fields(self):
        metas = [{"breadcrumb": "Informatika › Upis 2026/27"},
                 {"url": "https://imi.pmf.kg.ac.rs/oas-informatika"}]
        self.assertEqual(find_rank({"breadcrumb": ["Upis 2026/27"]}, ["", ""], metas), 1)
        self.assertEqual(find_rank({"url": ["imi.pmf.kg.ac.rs/oas-informatika"]}, ["", ""], metas), 2)

    def test_no_match(self):
        self.assertIsNone(find_rank({"text": ["dekan"]}, ["Upis."], [{}]))


class EvaluateTests(unittest.TestCase):
    def test_hits_and_mrr(self):
        result = evaluate(FakeStore(hybrid_helps_reranker_breaks), CASES, k=5)
        self.assertEqual(result["ranks"], [3, None])
        self.assertEqual(result["hits"], 1)
        self.assertAlmostEqual(result["mrr"], (1 / 3) / 2)


class CompareTests(unittest.TestCase):
    def test_every_variant_is_scored_and_flags_are_restored(self):
        store = FakeStore(hybrid_helps_reranker_breaks)
        store.use_synonyms = True  # as configured before the comparison
        lines = []

        print_comparison(store, CASES, k=5, out=lines.append)

        self.assertEqual(
            {flags for flags in store.searched_flags},
            {(False, False, False), (True, False, False), (False, True, False),
             (False, False, True), (True, True, True)},
        )
        self.assertEqual((store.use_synonyms, store.use_hybrid, store.use_reranker),
                         (True, False, False))
        # per question: baseline, synonyms and hybrid ranks
        self.assertTrue(any(re.search(r"\b3\s+3\s+1\s+Koliko modula", line) for line in lines))
        summary = {line.split()[0]: line for line in lines[-5:]}
        self.assertIn("1/2", summary["baseline"])
        self.assertIn("0.167", summary["baseline"])
        self.assertIn("0.500", summary["hybrid"])
        self.assertIn("failed: model download failed", summary["reranker"])
        self.assertIn("failed: model download failed", summary["all"])


@unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML not installed")
class LoadCasesTests(unittest.TestCase):
    def test_scraper_question_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval_questions.yaml"
            path.write_text(
                "- question: Koliko modula ima informatika?\n"
                "  expect:\n"
                "    text: [dva izborna modula]\n",
                encoding="utf-8",
            )
            self.assertEqual(load_cases(path), [CASES[0]])


if __name__ == "__main__":
    unittest.main()
