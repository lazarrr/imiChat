"""
reranker.py
~~~~~~~~~~~
Cross-encoder reranking (RERANKER_ENABLED).  Vector search compares a
question and a chunk only through their separate embeddings; a
cross-encoder reads both together and scores how well the chunk answers
the question (0–1), which orders the candidates much more accurately.

The default BAAI/bge-reranker-v2-m3 is multilingual (Serbian included) and
is downloaded on first use (~2.3 GB).  On CPU, scoring the
RETRIEVAL_FETCH_K candidates can take a few seconds per question — set
RERANKER_DEVICE=cuda if there is a GPU, or lower RETRIEVAL_FETCH_K.
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


class Reranker:
    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 512):
        # Imported here so this module (and keep_best) load without the
        # model stack when reranking is switched off.
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=max_length, device=device)

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Relevance of each document to the query (0–1, higher is better)."""
        if not documents:
            return []
        scores = self.model.predict(
            [(query, doc) for doc in documents], show_progress_bar=False
        )
        return [float(s) for s in scores]


def keep_best(
    scores: list[float], items: list[T], k: int, min_score: float = 0.0
) -> list[tuple[float, T]]:
    """The ``k`` best-scoring items that score at least ``min_score``, best first."""
    ranked = sorted(zip(scores, items), key=lambda pair: pair[0], reverse=True)
    return [pair for pair in ranked if pair[0] >= min_score][:k]
