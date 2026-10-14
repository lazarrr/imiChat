"""Unit tests for BM25, RRF fusion and deduplication."""

from src.chains.retriever import HybridRetriever, RetrievedChunk
from src.core.bm25 import BM25Index, tokenize


class _Config:
    HYBRID_RETRIEVAL_ENABLED = True
    RETRIEVAL_CANDIDATES = 10
    RRF_K = 60
    RERANK_ENABLED = False
    BM25_CACHE_PATH = "/tmp/unused-bm25-cache.pkl"


def test_tokenize_normalizes_serbian_diacritics():
    assert tokenize("Čačak, Đakovo i Šabac") == ["cacak", "djakovo", "i", "sabac"]


def test_bm25_ranks_exact_terms_first():
    index = BM25Index(
        ids=["1", "2", "3"],
        documents=[
            "Projekat 37013 se bavi održavanjem visokih brana u Srbiji.",
            "Studijski program informatike traje četiri godine.",
            "Raspored ispita za prvi semestar.",
        ],
        metadatas=[{}, {}, {}],
    )

    hits = index.search("projekat 37013 visokih brana", k=3)

    assert hits, "expected BM25 hits"
    assert hits[0][0] == 0
    assert index.search("nepostojeci termin", k=3) == []


def test_rrf_fusion_rewards_chunks_ranked_by_both_retrievers():
    retriever = HybridRetriever(vector_store=object(), config=_Config())
    dense = [
        RetrievedChunk(id="a", document="a", dense_rank=0),
        RetrievedChunk(id="b", document="b", dense_rank=1),
    ]
    bm25 = [
        RetrievedChunk(id="b", document="b", bm25_rank=0),
        RetrievedChunk(id="c", document="c", bm25_rank=1),
    ]

    fused = retriever._fuse(dense, bm25)
    scores = {chunk.id: chunk.score for chunk in fused}

    assert scores["b"] > scores["a"] > scores["c"] > 0
    assert [chunk.id for chunk in fused][0] == "b"


def test_dedupe_drops_identical_text():
    retriever = HybridRetriever(vector_store=object(), config=_Config())
    chunks = [
        RetrievedChunk(id="a", document="isti   tekst"),
        RetrievedChunk(id="b", document="isti tekst"),
        RetrievedChunk(id="c", document="drugi tekst"),
    ]

    unique = retriever._dedupe(chunks)

    assert [chunk.id for chunk in unique] == ["a", "c"]


def test_retriever_skips_bm25_when_hybrid_disabled():
    class _DenseOnlyConfig(_Config):
        HYBRID_RETRIEVAL_ENABLED = False

    class _FakeVectorStore:
        @staticmethod
        def similarity_search(query, k):
            return {
                "ids": [["x"]],
                "documents": [["tekst"]],
                "metadatas": [[{"source": "a.pdf", "filename": "a.pdf"}]],
            }

    retriever = HybridRetriever(
        vector_store=_FakeVectorStore(), config=_DenseOnlyConfig()
    )
    chunks = retriever.retrieve("upit", k=5)

    assert retriever._bm25 is None
    assert [chunk.id for chunk in chunks] == ["x"]
    assert chunks[0].dense_similarity is None


def test_dense_search_reports_cosine_similarity():
    class _DenseOnlyConfig(_Config):
        HYBRID_RETRIEVAL_ENABLED = False

    class _FakeVectorStore:
        @staticmethod
        def similarity_search(query, k):
            return {
                "ids": [["x", "y"]],
                "documents": [["prvi", "drugi"]],
                "metadatas": [
                    [{"source": "a.pdf"}, {"source": "b.pdf"}]
                ],
                "distances": [[0.12, 0.5]],
            }

    retriever = HybridRetriever(
        vector_store=_FakeVectorStore(), config=_DenseOnlyConfig()
    )
    chunks = retriever.retrieve("upit", k=5)

    assert abs(chunks[0].dense_similarity - 0.88) < 1e-9
    assert abs(chunks[1].dense_similarity - 0.5) < 1e-9
