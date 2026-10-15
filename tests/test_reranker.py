"""
Tests for picking the best reranked chunks (src/core/reranker.py).
The cross-encoder itself is exercised through tests/test_vector_store.py
with a fake model.

    python -m unittest tests.test_reranker -v
"""

import unittest

from src.core.reranker import keep_best


class KeepBestTests(unittest.TestCase):
    def test_best_first_limited_to_k(self):
        self.assertEqual(
            keep_best([0.2, 0.9, 0.5], ["a", "b", "c"], k=2),
            [(0.9, "b"), (0.5, "c")],
        )

    def test_min_score_drops_weak_items(self):
        self.assertEqual(
            keep_best([0.2, 0.9, 0.05], ["a", "b", "c"], k=5, min_score=0.1),
            [(0.9, "b"), (0.2, "a")],
        )


if __name__ == "__main__":
    unittest.main()
