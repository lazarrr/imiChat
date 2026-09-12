"""
retriever.py
~~~~~~~~~~~~
Hybrid retrieval: dense E5 embeddings + lexical BM25, fused with
Reciprocal Rank Fusion (RRF).

Why: dense retrieval generalises well but blurs exact identifiers
(course codes, acronyms, project numbers).  BM25 is exact-match strong
but misses paraphrases.  Fusing both rankings keeps the strengths of
each and materially improves questions that mention a concrete course
or acronym.

Optionally a multilingual cross-encoder (``mmarco-mMiniLMv2``) reranks
the fused candidates.  It is small enough for CPU inference and is
disabled unless ``RERANK_ENABLED`` is set.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field

from src.core.bm25 import BM25Index, load_or_build_bm25

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class RetrievedChunk:
    """A single retrieved chunk plus its ranking provenance."""

    id: str
    document: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None


def _stable_id(document: str) -> str:
    return hashlib.md5(document.encode("utf-8")).hexdigest()


class HybridRetriever:
    def __init__(self, vector_store, config=None) -> None:
        self.vector_store = vector_store
        self.config = config

        self.hybrid_enabled = bool(
            getattr(config, "HYBRID_RETRIEVAL_ENABLED", True)
        )
        self.candidates = int(getattr(config, "RETRIEVAL_CANDIDATES", 30))
        self.rrf_k = int(getattr(config, "RRF_K", 60))
        # Dense stays the primary signal; BM25 gets a light weight because
        # the crawled corpus repeats page navigation on almost every chunk,
        # which inflates lexical scores for boilerplate-heavy pages.  The
        # light weight is enough to break ties in favour of exact course /
        # acronym / name matches without overriding dense semantics.
        self.bm25_weight = float(getattr(config, "BM25_FUSION_WEIGHT", 0.1))

        self.rerank_enabled = bool(getattr(config, "RERANK_ENABLED", False))
        self.rerank_model_name = getattr(
            config, "RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        )
        self.rerank_candidates = int(getattr(config, "RERANK_CANDIDATES", 20))
        self.bm25_cache_path = getattr(
            config, "BM25_CACHE_PATH", "./data/vector_store/bm25_cache.pkl"
        )

        self._bm25: BM25Index | None = None
        self._reranker = None

    # ------------------------------------------------------------------ #
    #  Dense / lexical backends                                            #
    # ------------------------------------------------------------------ #

    def _dense_search(self, query: str, k: int) -> list[RetrievedChunk]:
        results = self.vector_store.similarity_search(query, k=k)
        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for rank, (document, metadata) in enumerate(zip(documents, metadatas)):
            chunk_id = ids[rank] if rank < len(ids) else _stable_id(document)
            chunks.append(
                RetrievedChunk(
                    id=chunk_id,
                    document=document,
                    metadata=metadata or {},
                    dense_rank=rank,
                )
            )
        return chunks

    def _bm25_index(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = load_or_build_bm25(
                self.vector_store, self.bm25_cache_path
            )
        return self._bm25

    # ------------------------------------------------------------------ #
    #  Fusion                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _reciprocal_rank(rank: int | None, rrf_k: int) -> float:
        if rank is None:
            return 0.0
        return 1.0 / (rrf_k + rank + 1)

    def _fuse(
        self,
        dense: list[RetrievedChunk],
        bm25: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion keyed by chunk id."""
        merged: dict[str, RetrievedChunk] = {}
        for chunk in dense:
            merged[chunk.id] = chunk
        for chunk in bm25:
            if chunk.id in merged:
                # Keep the dense object (it carries the document/metadata)
                # but record the lexical rank so both contribute to RRF.
                merged[chunk.id].bm25_rank = chunk.bm25_rank
            else:
                merged[chunk.id] = chunk

        for chunk in merged.values():
            chunk.score = 1.0 * self._reciprocal_rank(
                chunk.dense_rank, self.rrf_k
            ) + self.bm25_weight * self._reciprocal_rank(
                chunk.bm25_rank, self.rrf_k
            )

        return sorted(merged.values(), key=lambda c: c.score, reverse=True)

    @staticmethod
    def _dedupe(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Drop exact duplicate texts (ChromaDB ids are content hashes, but
        BM25 and dense can still surface the same text through re-ingestion)."""
        seen_ids: set[str] = set()
        seen_text: set[str] = set()
        unique: list[RetrievedChunk] = []
        for chunk in chunks:
            text_key = _WHITESPACE_RE.sub(" ", chunk.document).strip().lower()
            if chunk.id in seen_ids or text_key in seen_text:
                continue
            seen_ids.add(chunk.id)
            seen_text.add(text_key)
            unique.append(chunk)
        return unique

    # ------------------------------------------------------------------ #
    #  Optional cross-encoder rerank                                       #
    # ------------------------------------------------------------------ #

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            device = getattr(self.config, "EMBEDDING_DEVICE", "cpu")
            logger.info("Loading reranker model %s (%s)", self.rerank_model_name, device)
            self._reranker = CrossEncoder(self.rerank_model_name, device=device)
        return self._reranker

    def _rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        reranker = self._get_reranker()
        pairs = [(query, chunk.document) for chunk in chunks]
        scores = reranker.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """
        Retrieve the ``k`` best chunks for ``query``.

        Disabled hybrid mode degenerates to pure dense retrieval;
        disabled reranking keeps the RRF order.  Everything is deduplicated
        before it is returned.
        """
        dense = self._dense_search(query, k=max(k, self.candidates))

        if not self.hybrid_enabled:
            return self._dedupe(dense)[:k]

        bm25_index = self._bm25_index()
        bm25_hits = bm25_index.search(query, k=max(k, self.candidates))
        bm25_chunks: list[RetrievedChunk] = []
        for rank, (corpus_idx, score) in enumerate(bm25_hits):
            bm25_chunks.append(
                RetrievedChunk(
                    id=bm25_index.ids[corpus_idx],
                    document=bm25_index.documents[corpus_idx],
                    metadata=bm25_index.metadatas[corpus_idx] or {},
                    score=score,
                    bm25_rank=rank,
                )
            )

        fused = self._dedupe(self._fuse(dense, bm25_chunks))
        logger.info(
            "Hybrid retrieval: %d dense + %d bm25 → %d fused",
            len(dense),
            len(bm25_chunks),
            len(fused),
        )

        if not self.rerank_enabled:
            return fused[:k]

        candidates = fused[: max(k, self.rerank_candidates)]
        return self._dedupe(self._rerank(query, candidates))[:k]
