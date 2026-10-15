"""
Tests for VectorStoreManager.similarity_search with the optional retrieval
steps switched on and off (src/core/vector_store.py).  Chroma, the E5
model and the reranker are replaced by fakes, but the module itself needs
the project's dependencies (chromadb, sentence-transformers, ...), so the
tests are skipped where those are not installed.

    python -m unittest tests.test_vector_store -v
"""

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

try:
    from src.core import vector_store
except ImportError:  # chromadb / sentence-transformers / ... not installed
    vector_store = None

CHUNKS = {
    "v1": ("Informatika ima dva izborna modula.", {}),
    "old": ("Osnovi programiranja po starom programu nose 8 ESPB.", {"status": "legacy"}),
    "v2": ("Upis na master studije počinje u julu.", {}),
    "v3": ("Opšte informacije o fakultetu.", {}),
    "k1": ("Predmet Osnovi programiranja nosi 6 ESPB.", {}),  # vector search misses it
}
VECTOR_HITS = [("v1", 0.10), ("old", 0.12), ("v2", 0.20), ("v3", 0.25)]


class FakeCollection:
    def __init__(self, chunks, vector_hits):
        self.chunks = dict(chunks)
        self.vector_hits = vector_hits
        self.get_calls = 0

    def query(self, query_embeddings, n_results, where, include):
        hits = [
            (chunk_id, dist) for chunk_id, dist in self.vector_hits
            if where is None or vector_store._is_current(self.chunks[chunk_id][1])
        ][:n_results]
        return {
            "ids": [[c for c, _ in hits]],
            "documents": [[self.chunks[c][0] for c, _ in hits]],
            "metadatas": [[self.chunks[c][1] for c, _ in hits]],
            "distances": [[d for _, d in hits]],
        }

    def count(self):
        return len(self.chunks)

    def get(self, include):
        self.get_calls += 1
        ids = list(self.chunks)
        return {
            "ids": ids,
            "documents": [self.chunks[c][0] for c in ids],
            "metadatas": [self.chunks[c][1] for c in ids],
        }


class FakeVector(list):
    def tolist(self):
        return list(self)


class FakeEmbeddingModel:
    def __init__(self):
        self.texts = []

    def encode(self, texts, **kwargs):
        self.texts.extend(texts)
        return [FakeVector([0.0]) for _ in texts]


class FakeReranker:
    SCORES = {
        CHUNKS["v3"][0]: 0.9,
        CHUNKS["v1"][0]: 0.6,
        CHUNKS["v2"][0]: 0.1,
    }

    def __init__(self, model_name, device="cpu"):
        self.queries = []

    def score(self, query, documents):
        self.queries.append(query)
        return [self.SCORES.get(doc, 0.0) for doc in documents]


def make_store(collection, **settings):
    config = SimpleNamespace(**{
        "EMBEDDING_MODEL": "e5", "EMBEDDING_DEVICE": "cpu", "VECTOR_DB_PATH": "unused",
        "CHROMA_COLLECTION": "test", "TOP_K_RESULTS": 2, "RETRIEVAL_FETCH_K": 4,
        "CHUNK_SIZE": 1000, "CHUNK_OVERLAP": 200, "SIMILARITY_THRESHOLD": 0.7,
        "QUERY_SYNONYMS": False, "HYBRID_SEARCH": False, "RERANKER_ENABLED": False,
        "RERANKER_MIN_SCORE": 0.0, "SYNONYMS_FILE": "config/synonyms.txt",
        **settings,
    })
    model = FakeEmbeddingModel()
    with patch.object(vector_store, "SentenceTransformer", return_value=model), \
         patch.object(vector_store.chromadb, "PersistentClient") as client, \
         patch.object(vector_store, "DocumentProcessor"), \
         patch.object(vector_store, "Reranker", FakeReranker):
        client.return_value.get_collection.return_value = collection
        store = vector_store.VectorStoreManager(config)
    return store, model


def documents(results):
    return results["documents"][0]


@unittest.skipIf(vector_store is None, "chromadb / sentence-transformers not installed")
class SimilaritySearchTests(unittest.TestCase):
    def setUp(self):
        self.collection = FakeCollection(CHUNKS, VECTOR_HITS)

    def test_all_steps_off_is_the_plain_vector_search(self):
        store, model = make_store(self.collection)

        results = store.similarity_search("Koliko modula ima informatika?")

        self.assertEqual(documents(results), [CHUNKS["v1"][0], CHUNKS["v2"][0]])
        self.assertEqual(model.texts, ["query: Koliko modula ima informatika?"])
        self.assertEqual(self.collection.get_calls, 0)  # no keyword index built

    def test_hybrid_adds_chunks_only_keyword_search_finds(self):
        store, _ = make_store(self.collection, HYBRID_SEARCH=True)

        results = store.similarity_search("Koliko ESPB nosi Osnovi programiranja?")

        self.assertIn(CHUNKS["k1"][0], documents(results))
        self.assertNotIn(CHUNKS["old"][0], documents(results))  # legacy stays out

    def test_hybrid_keeps_legacy_chunks_when_asked_about_the_old_program(self):
        store, _ = make_store(self.collection, HYBRID_SEARCH=True)

        results = store.similarity_search("Koliko ESPB je imao Osnovi programiranja po starom programu?")

        self.assertIn(CHUNKS["old"][0], documents(results))

    def test_keyword_index_is_rebuilt_when_the_collection_changes(self):
        store, _ = make_store(self.collection, HYBRID_SEARCH=True)
        store.similarity_search("ESPB")
        store.similarity_search("ESPB")
        self.assertEqual(self.collection.get_calls, 1)

        self.collection.chunks["k2"] = ("Novi predmet, 4 ESPB.", {})
        store.similarity_search("ESPB")
        self.assertEqual(self.collection.get_calls, 2)

    def test_reranker_reorders_and_drops_low_scores(self):
        store, _ = make_store(self.collection, RERANKER_ENABLED=True, RERANKER_MIN_SCORE=0.5)

        results = store.similarity_search("Koliko modula ima informatika?")

        # candidates v1, v2, v3 scored 0.6, 0.1, 0.9 → best two above 0.5
        self.assertEqual(documents(results), [CHUNKS["v3"][0], CHUNKS["v1"][0]])

    def test_synonyms_expand_the_search_but_not_the_reranker_question(self):
        store, model = make_store(self.collection, QUERY_SYNONYMS=True, RERANKER_ENABLED=True)

        store.similarity_search("Koji su smerovi na informatici?")

        self.assertEqual(model.texts, ["query: Koji su smerovi na informatici? (modul)"])
        self.assertEqual(store._reranker.queries, ["Koji su smerovi na informatici?"])

    def test_steps_can_be_switched_at_runtime(self):
        store, _ = make_store(self.collection)
        store.use_hybrid = True

        results = store.similarity_search("Koliko ESPB nosi Osnovi programiranja?")

        self.assertIn(CHUNKS["k1"][0], documents(results))

    def test_searches_from_several_threads_run_one_at_a_time(self):
        # The API searches from worker threads; the models and the lazily
        # built keyword index must never serve two searches at once.
        store, model = make_store(self.collection, HYBRID_SEARCH=True)
        encode = model.encode
        counter_lock = threading.Lock()
        running = peak = 0

        def slow_encode(texts, **kwargs):
            nonlocal running, peak
            with counter_lock:
                running += 1
                peak = max(peak, running)
            time.sleep(0.05)  # room for a second search to slip in
            with counter_lock:
                running -= 1
            return encode(texts, **kwargs)

        model.encode = slow_encode
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(store.similarity_search, ["ESPB"] * 4))

        self.assertEqual(peak, 1)
        self.assertEqual(self.collection.get_calls, 1)  # keyword index built once


if __name__ == "__main__":
    unittest.main()
