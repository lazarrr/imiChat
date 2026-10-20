"""
Tests for the keyword side of retrieval (src/core/keyword_search.py).

    python -m unittest tests.test_keyword_search -v
"""

import tempfile
import unittest
from pathlib import Path

from src.core.keyword_search import (
    BM25Index,
    SynonymExpander,
    fold,
    reciprocal_rank_fusion,
    tokenize,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TokenizeTests(unittest.TestCase):
    def test_fold_removes_case_diacritics_and_cyrillic(self):
        self.assertEqual(fold("Šta je ŽIVOT, đače?"), "sta je zivot, djace?")
        self.assertEqual(fold("Шта је ЕСПБ"), "sta je espb")

    def test_inflected_forms_share_a_stem(self):
        for forms in (
            ["predmet", "predmeta", "predmetu", "predmetima"],
            ["modul", "modula", "moduli", "modulima"],
            ["smer", "smera", "smerovi", "smerova"],
            ["studije", "studija", "studijama"],
            ["upis", "upisa", "upisu"],
        ):
            with self.subTest(forms=forms):
                self.assertEqual(len({tokenize(form)[0] for form in forms}), 1)

    def test_stopwords_are_dropped(self):
        self.assertEqual(
            tokenize("Koliko ESPB ima Uvod u programiranje?"),
            ["espb", "uvod", "programiranj"],
        )


class BM25IndexTests(unittest.TestCase):
    def setUp(self):
        self.index = BM25Index(
            ["c1", "c2", "c3", "old"],
            [
                "Predmet Osnovi programiranja nosi 6 ESPB bodova.",
                "Upis na master studije počinje u julu.",
                "Raspored časova za prvu godinu informatike.",
                "Predmet Osnovi programiranja po starom programu nosi 8 ESPB.",
            ],
            [{}, {}, {}, {"status": "legacy"}],
        )

    def test_only_chunks_sharing_words_are_found(self):
        hits = self.index.search("ESPB za Osnovi programiranja", n=10)
        self.assertEqual({chunk_id for chunk_id, _, _ in hits}, {"c1", "old"})

    def test_allow_filters_by_metadata(self):
        hits = self.index.search(
            "ESPB za Osnovi programiranja", n=10,
            allow=lambda meta: meta.get("status") != "legacy",
        )
        self.assertEqual([chunk_id for chunk_id, _, _ in hits], ["c1"])

    def test_more_occurrences_rank_higher(self):
        index = BM25Index(["a", "b"], ["ESPB ESPB predmet", "ESPB predmet raspored"], [{}, {}])
        self.assertEqual([chunk_id for chunk_id, _, _ in index.search("ESPB", n=2)], ["a", "b"])

    def test_n_limits_hits_and_no_overlap_finds_nothing(self):
        self.assertEqual(len(self.index.search("ESPB programiranja", n=1)), 1)
        self.assertEqual(self.index.search("dekan fakulteta", n=5), [])


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_items_ranked_well_by_both_lists_come_first(self):
        self.assertEqual(
            reciprocal_rank_fusion(["a", "b", "c"], ["c", "a", "d"]),
            ["a", "c", "b", "d"],
        )


class SynonymExpanderTests(unittest.TestCase):
    def setUp(self):
        self.expander = SynonymExpander([
            ["smer", "modul"],
            ["OAS", "osnovne akademske studije"],
        ])

    def test_student_word_adds_the_official_term(self):
        self.assertEqual(
            self.expander.expand("Koji su smerovi na informatici?"),
            "Koji su smerovi na informatici? (modul)",
        )

    def test_abbreviations_and_phrases_expand_both_ways(self):
        self.assertEqual(
            self.expander.expand("Koliko traje OAS?"),
            "Koliko traje OAS? (osnovne akademske studije)",
        )
        self.assertEqual(
            self.expander.expand("Koliko traju osnovne akademske studije?"),
            "Koliko traju osnovne akademske studije? (OAS)",
        )

    def test_unrelated_or_already_complete_queries_are_unchanged(self):
        for query in ("Ko je dekan fakulteta?", "Razlika između smera i modula?"):
            with self.subTest(query=query):
                self.assertEqual(self.expander.expand(query), query)

    def test_file_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synonyms.txt"
            path.write_text(
                "# comment\n\nsmer, modul   # inline comment\nsingle term\n",
                encoding="utf-8",
            )
            expander = SynonymExpander.from_file(path)
        self.assertEqual([[term for term, _ in group] for group in expander.groups],
                         [["smer", "modul"]])

    def test_project_synonyms_file_loads(self):
        expander = SynonymExpander.from_file(PROJECT_ROOT / "config" / "synonyms.txt")
        self.assertEqual(expander.expand("Koji su smerovi?"), "Koji su smerovi? (modul)")


if __name__ == "__main__":
    unittest.main()
