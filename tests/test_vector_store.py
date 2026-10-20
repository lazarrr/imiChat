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


def _where_matches(meta, where):
    """Minimal Chroma metadata-filter evaluator for the fake collection."""
    if where is None:
        return True
    if "$and" in where:
        return all(_where_matches(meta, item) for item in where["$and"])
    if "$or" in where:
        return any(_where_matches(meta, item) for item in where["$or"])
    for key, condition in where.items():
        value = (meta or {}).get(key)
        if not isinstance(condition, dict):
            if value != condition:
                return False
            continue
        for operator, operand in condition.items():
            if operator == "$eq" and value != operand:
                return False
            if operator == "$ne" and key in (meta or {}) and value == operand:
                return False
            if operator == "$in" and value not in operand:
                return False
            if operator == "$nin" and value in operand:
                return False
    return True


class FakeCollection:
    def __init__(self, chunks, vector_hits):
        self.chunks = dict(chunks)
        self.vector_hits = vector_hits
        self.get_calls = 0
        self.query_where_log = []

    def query(self, query_embeddings, n_results, where, include):
        self.query_where_log.append(where)
        hits = [
            (chunk_id, dist) for chunk_id, dist in self.vector_hits
            if _where_matches(self.chunks[chunk_id][1], where)
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
        "CHROMA_COLLECTION": "test", "TOP_K_RESULTS": 2, "MAX_CONTEXT_CHUNKS": 4,
        "RETRIEVAL_FETCH_K": 4,
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

    def test_diagnostics_are_off_by_default(self):
        store, _ = make_store(self.collection)

        results = store.similarity_search("Koliko modula ima informatika?")

        # No extra query, no extra keys: normal searches pay no price.
        self.assertNotIn("excluded", results)
        self.assertEqual(self.collection.query_where_log, [vector_store._CURRENT_ONLY_WHERE])

    def test_diagnostics_report_the_chunks_below_the_threshold(self):
        # SIMILARITY_THRESHOLD 0.85 -> max distance 0.15: v1 (0.10) clears it,
        # v2 (0.20) and v3 (0.25) do not.
        store, _ = make_store(self.collection, SIMILARITY_THRESHOLD=0.85)

        results = store.similarity_search("Koliko modula ima informatika?", diagnostics=True)

        excluded = results["excluded"]
        self.assertEqual(results["excluded_total"], len(excluded))
        dropped = {c["chunk_id"]: c for c in excluded if c["reason"] == "excluded_threshold"}
        # The fake store ignores `where` on the wide query, so v1 may show up
        # here repeating a visible chunk; what matters is that the
        # over-threshold chunks are reported with their distance and the
        # threshold that dropped them.
        self.assertEqual(sorted(dropped), ["v2", "v3"])
        # 1.0 - 0.85 is 0.15000000000000002 in binary floating point
        self.assertAlmostEqual(dropped["v3"]["threshold"], 0.15)
        self.assertEqual(dropped["v3"]["distance"], 0.25)
        # The visible result is what the threshold allows, unchanged by
        # diagnostics being on.
        self.assertEqual(documents(results), [CHUNKS["v1"][0]])

    def test_diagnostics_report_chunks_the_version_policy_dropped(self):
        store, _ = make_store(self.collection)

        results = store.similarity_search("Koliko modula ima informatika?", diagnostics=True)

        legacy = [c for c in results["excluded"] if c["reason"] == "excluded_legacy_version"]
        self.assertTrue(legacy)
        self.assertEqual(legacy[0]["metadata"]["status"], "legacy")

    def test_diagnostics_report_the_reranker_cut(self):
        store, _ = make_store(self.collection, RERANKER_ENABLED=True, RERANKER_MIN_SCORE=0.5)

        results = store.similarity_search("Koliko modula ima informatika?", diagnostics=True)

        dropped = [c for c in results["excluded"] if c["reason"] == "excluded_reranker"]
        # v2 scored 0.1, below RERANKER_MIN_SCORE
        self.assertEqual([c["text"] for c in dropped], [CHUNKS["v2"][0]])
        self.assertEqual(dropped[0]["rerank_score"], 0.1)
        self.assertEqual(dropped[0]["rerank_min_score"], 0.5)

    def test_a_question_about_the_old_program_skips_the_current_version_filter(self):
        store, _ = make_store(self.collection)

        results = store.similarity_search(
            "Koliko ESPB je imao Osnovi programiranja po starom programu?", diagnostics=True)

        # The old-program filter is applied inside Chroma (instead of the
        # current-only filter); diagnostics then query without it to report
        # what the old-program filter hid.
        current_where = self.collection.query_where_log[0]
        self.assertNotEqual(current_where, vector_store._CURRENT_ONLY_WHERE)
        self.assertEqual(self.collection.query_where_log[1], None)
        self.assertEqual(
            [c for c in results["excluded"] if c["reason"] == "excluded_legacy_version"], [])

    def test_a_failing_diagnostic_query_does_not_break_the_search(self):
        store, _ = make_store(self.collection)  # default threshold: v1, v2 kept
        original = self.collection.query
        calls = {"n": 0}

        def flaky(query_embeddings, n_results, where, include):
            calls["n"] += 1
            if calls["n"] == 2:  # the diagnostics query
                raise RuntimeError("chroma hiccup")
            return original(query_embeddings, n_results, where, include)

        self.collection.query = flaky
        with self.assertLogs("src.core.vector_store", level="WARNING"):
            results = store.similarity_search("Koliko modula ima informatika?", diagnostics=True)

        # The failed diagnostic query costs the exclusion list, not the
        # answer: the visible result is the normal one (v1 and v2 at the
        # default threshold).
        self.assertEqual(documents(results), [CHUNKS["v1"][0], CHUNKS["v2"][0]])
        self.assertIn("excluded", results)

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


VERSION_CHUNKS = {
    "neutral": ("Opšte informacije o fakultetu.", {}),
    "current": (
        "Informatika 2026 ima dva izborna modula.",
        {"status": "enrolling", "accreditation_year": 2026},
    ),
    "old2019": (
        "Informatika 2019 ima tri izborna modula.",
        {"status": "phase_out", "accreditation_year": 2019,
         "valid_for": "studenti upisani 2019/20–2025/26"},
    ),
    "old2011": (
        "Informatika 2011 ima pet izbornih modula.",
        {"status": "legacy", "accreditation_year": 2011},
    ),
    "upcoming2027": (
        "Matematika 2027 ima tri modula.",
        {"status": "upcoming", "accreditation_year": 2027,
         "valid_for": "upis od 2027/28"},
    ),
    "outdated": ("FAQ: informatika ima tri smera.", {"outdated": True}),
}
VERSION_HITS = [
    ("current", 0.10), ("old2019", 0.11), ("upcoming2027", 0.12),
    ("neutral", 0.13), ("old2011", 0.14), ("outdated", 0.08),
]


@unittest.skipIf(vector_store is None, "chromadb / sentence-transformers not installed")
class VersionAwareSearchTests(unittest.TestCase):
    def setUp(self):
        self.collection = FakeCollection(VERSION_CHUNKS, VERSION_HITS)

    def test_default_question_uses_the_current_version_only(self):
        store, _ = make_store(
            self.collection, TOP_K_RESULTS=3, RETRIEVAL_FETCH_K=10)

        results = store.similarity_search("Koliko modula ima informatika?")

        # enrolling + version-agnostic; phase_out, legacy and upcoming are out
        self.assertEqual(
            documents(results),
            [VERSION_CHUNKS["current"][0], VERSION_CHUNKS["neutral"][0]],
        )
        self.assertEqual(
            self.collection.query_where_log, [vector_store._CURRENT_ONLY_WHERE])

    def test_a_question_about_an_old_year_fetches_that_version(self):
        store, _ = make_store(
            self.collection, TOP_K_RESULTS=3, RETRIEVAL_FETCH_K=10)

        results = store.similarity_search(
            "Koliko modula je imala informatika 2019. godine?")

        self.assertEqual(documents(results), [VERSION_CHUNKS["old2019"][0]])
        self.assertEqual(self.collection.query_where_log[0], {
            "$and": [
                {"accreditation_year": {"$in": [2019]}},
                {"outdated": {"$ne": True}},
            ]
        })

    def test_a_question_about_a_future_year_fetches_the_upcoming_version(self):
        store, _ = make_store(
            self.collection, TOP_K_RESULTS=3, RETRIEVAL_FETCH_K=10)

        results = store.similarity_search(
            "Koji moduli postoje za upis 2027/28?")

        self.assertEqual(documents(results), [VERSION_CHUNKS["upcoming2027"][0]])

    def test_a_generic_old_program_question_fetches_legacy_and_phase_out(self):
        store, _ = make_store(
            self.collection, TOP_K_RESULTS=3, RETRIEVAL_FETCH_K=10)

        results = store.similarity_search(
            "Šta piše u starom programu informatike?")

        self.assertEqual(
            documents(results),
            [VERSION_CHUNKS["old2019"][0], VERSION_CHUNKS["old2011"][0]],
        )

    def test_an_enrollment_year_matches_the_versions_valid_for_range(self):
        store, _ = make_store(
            self.collection, TOP_K_RESULTS=3, RETRIEVAL_FETCH_K=10)

        results = store.similarity_search(
            "Koji program je važio za upis 2025. godine?")

        # 2025 is not an accreditation year: the exact-year query finds
        # nothing, the phase_out version's valid_for range admits it.
        self.assertEqual(documents(results), [VERSION_CHUNKS["old2019"][0]])

    def test_an_index_without_version_metadata_is_still_answerable(self):
        unversioned = {
            f"u{i}": (f"Opšti podatak {i}.", {}) for i in range(4)
        }
        collection = FakeCollection(
            unversioned, [(chunk_id, 0.1 + i / 100) for i, chunk_id in enumerate(unversioned)])
        store, _ = make_store(collection, TOP_K_RESULTS=2, RETRIEVAL_FETCH_K=10)

        results = store.similarity_search("Šta je bilo na programu 2019. godine?")

        self.assertEqual(len(documents(results)), 2)

    def test_diagnostics_name_the_upcoming_status_and_always_drop_outdated(self):
        store, _ = make_store(self.collection)

        results = store.similarity_search(
            "Koliko modula ima informatika?", diagnostics=True)

        reasons = {
            chunk["metadata"].get("status") or "outdated": chunk["reason"]
            for chunk in results["excluded"]
        }
        self.assertEqual(reasons["phase_out"], "excluded_legacy_version")
        self.assertEqual(reasons["legacy"], "excluded_legacy_version")
        self.assertEqual(reasons["upcoming"], "excluded_upcoming_version")
        self.assertEqual(reasons["outdated"], "excluded_outdated")


@unittest.skipIf(vector_store is None, "chromadb / sentence-transformers not installed")
class ChunkLimitTests(unittest.TestCase):
    def test_the_model_never_gets_five_chunks(self):
        chunks = {f"n{i}": (f"Podatak {i}.", {}) for i in range(8)}
        hits = [(chunk_id, 0.1 + i / 100) for i, chunk_id in enumerate(chunks)]
        store, _ = make_store(
            FakeCollection(chunks, hits),
            TOP_K_RESULTS=10, MAX_CONTEXT_CHUNKS=4, RETRIEVAL_FETCH_K=20,
        )

        results = store.similarity_search("Sve informacije")

        self.assertEqual(len(documents(results)), 4)
        self.assertLess(len(documents(results)), 5)

    def test_an_explicit_smaller_k_is_honoured(self):
        chunks = {f"n{i}": (f"Podatak {i}.", {}) for i in range(8)}
        hits = [(chunk_id, 0.1 + i / 100) for i, chunk_id in enumerate(chunks)]
        store, _ = make_store(
            FakeCollection(chunks, hits),
            TOP_K_RESULTS=3, MAX_CONTEXT_CHUNKS=4, RETRIEVAL_FETCH_K=20,
        )

        self.assertEqual(len(documents(store.similarity_search("Sve", k=2))), 2)


if __name__ == "__main__":
    unittest.main()
