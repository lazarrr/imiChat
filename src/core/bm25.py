"""
bm25.py
~~~~~~~
Zero-dependency BM25Okapi implementation with an on-disk cache.

The hybrid retriever fuses this lexical ranking with the dense E5 ranking
via Reciprocal Rank Fusion.  Lexical matching is what makes exact course
names, acronyms and numbers ("MI111", "14th SMAK", "37013") survive,
which pure dense retrieval often blurs.

Serbian (Latin) text is normalised before tokenisation so that queries
written with or without diacritics match the same terms:

    "č ć ž š đ"  →  "c c z s dj"
"""

from __future__ import annotations

import hashlib
import logging
import math
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    """Lower-case and strip Serbian diacritics (đ first — it has no NFKD form)."""
    text = text.lower().replace("đ", "dj")
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    """Tokenize arbitrary text into normalised alphanumeric terms."""
    return _TOKEN_RE.findall(normalize_text(text))


def corpus_fingerprint(ids: list[str]) -> str:
    """Stable fingerprint of a corpus — content ids are content hashes,
    so any change to the corpus changes the fingerprint."""
    digest = hashlib.md5()
    for chunk_id in sorted(ids):
        digest.update(chunk_id.encode("utf-8"))
    digest.update(f"|{len(ids)}|".encode("utf-8"))
    return digest.hexdigest()


class BM25Index:
    """In-memory BM25Okapi index over an id/document/metadata corpus."""

    def __init__(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.ids = list(ids)
        self.documents = list(documents)
        self.metadatas = list(metadatas)
        self.k1 = k1
        self.b = b
        self.fingerprint = corpus_fingerprint(self.ids)

        self._tokens = [tokenize(doc) for doc in self.documents]
        self._doc_len = [len(tokens) for tokens in self._tokens]
        self._avgdl = sum(self._doc_len) / max(1, len(self._doc_len))

        doc_freq: Counter[str] = Counter()
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for idx, tokens in enumerate(self._tokens):
            term_freq = Counter(tokens)
            doc_freq.update(term_freq.keys())
            for term, freq in term_freq.items():
                self._postings[term].append((idx, freq))

        n_docs = max(1, len(self._tokens))
        self._idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    @property
    def size(self) -> int:
        return len(self.ids)

    def search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        """Return ``(corpus_index, score)`` pairs, best first."""
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for idx, term_freq in self._postings[term]:
                doc_len = self._doc_len[idx] or 1
                denominator = term_freq + self.k1 * (
                    1.0 - self.b + self.b * doc_len / self._avgdl
                )
                scores[idx] += idf * term_freq * (self.k1 + 1.0) / denominator

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:k]


def load_or_build_bm25(
    vector_store,
    cache_path: str | Path,
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Index:
    """Return a BM25 index for the vector store corpus, using the on-disk
    cache when its fingerprint matches and rebuilding it otherwise."""
    corpus = vector_store.fetch_corpus()
    ids = corpus["ids"]
    fingerprint = corpus_fingerprint(ids)

    cache_file = Path(cache_path)
    if cache_file.exists():
        try:
            with cache_file.open("rb") as handle:
                cached = pickle.load(handle)
            if (
                isinstance(cached, BM25Index)
                and cached.fingerprint == fingerprint
                and cached.size == len(ids)
            ):
                logger.info("BM25 index loaded from cache (%d chunks)", cached.size)
                return cached
            logger.info("BM25 cache is stale — rebuilding")
        except Exception as exc:  # corrupted cache must never break startup
            logger.warning("Could not load BM25 cache (%s) — rebuilding", exc)

    index = BM25Index(
        ids=ids,
        documents=corpus["documents"],
        metadatas=corpus["metadatas"],
        k1=k1,
        b=b,
    )
    logger.info("BM25 index built over %d chunks", index.size)

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("wb") as handle:
            pickle.dump(index, handle)
    except Exception as exc:
        logger.warning("Could not persist BM25 cache: %s", exc)

    return index
