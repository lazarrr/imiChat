"""
vector_store.py
~~~~~~~~~~~~~~~
Read-only ChromaDB vector store for university-chatbot.

Reads shared_config.json from the persist directory to discover the
embedding model, collection name, E5 prefixes, and HNSW space that the
university-scraper used when populating the store.  This guarantees
query-time embedding parameters match write-time exactly.

Capabilities (READ side only):
  • Load SentenceTransformer with the configured embedding model
  • Create query embeddings with the E5 "query: " prefix
  • Similarity search with threshold filtering
"""

from __future__ import annotations

import json
import os

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer


class VectorStoreManager:
    """
    Read-only vector store for the chatbot.

    All shared constants (model name, collection name, prefixes,
    HNSW space, normalisation) are read from shared_config.json
    in the persist directory.  If the file is missing the manager
    falls back to hard-coded defaults and logs a warning.
    """

    # ------------------------------------------------------------------ #
    #  Hard-coded fallback defaults                                        #
    #  (must match shared_config.json + university-scraper exactly)        #
    # ------------------------------------------------------------------ #
    _DEFAULTS: dict = {
        "collection_name": "university_knowledge",
        "embedding_model": "intfloat/multilingual-e5-large",
        "embedding_device": "cpu",
        "normalize_embeddings": True,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "hnsw_space": "cosine",
    }

    def __init__(self, config):
        self.config = config

        # ── Load shared config from the persist directory ─────────── #
        self._shared = self._load_shared_config(config.VECTOR_DB_PATH)

        # ── SentenceTransformer ───────────────────────────────────── #
        self.embeddings_model = SentenceTransformer(
            self._shared["embedding_model"],
            device=self._shared["embedding_device"],
        )
        self._normalize = self._shared["normalize_embeddings"]

        # ── ChromaDB client (read-only intent) ────────────────────── #
        self.client = chromadb.PersistentClient(
            path=config.VECTOR_DB_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection_name = self._shared["collection_name"]

        # Derived threshold: ChromaDB uses cosine distance (0=identical, 2=opposite)
        self._distance_threshold = 1.0 - config.SIMILARITY_THRESHOLD

    # ------------------------------------------------------------------ #
    #  Shared config                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def _load_shared_config(cls, persist_dir: str) -> dict:
        """Load shared_config.json, falling back to hard-coded defaults."""
        path = os.path.join(persist_dir, "shared_config.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                merged = {**cls._DEFAULTS, **loaded}
                print(f"Loaded shared config from {path}")
                return merged
            except (json.JSONDecodeError, OSError) as exc:
                print(f"Warning: could not parse {path}: {exc}.  Using defaults.")
        else:
            print(
                f"Warning: {path} not found.  "
                f"Make sure the university-scraper has been run at least once "
                f"to populate this directory and generate shared_config.json.  "
                f"Using hard-coded defaults for now."
            )
        return dict(cls._DEFAULTS)

    # ------------------------------------------------------------------ #
    #  Embeddings                                                          #
    # ------------------------------------------------------------------ #

    def create_embeddings(self, texts: list[str], is_query: bool = True) -> np.ndarray:
        """
        Encode texts with the correct E5 prefix.

        The scraper uses ``passage: `` when writing; the chatbot MUST
        use ``query: `` when searching.  This asymmetry is what makes
        E5-family models work correctly for retrieval.
        """
        prefix = self._shared["query_prefix"] if is_query else self._shared["passage_prefix"]
        prefixed = [prefix + t for t in texts]
        return self.embeddings_model.encode(
            prefixed,
            normalize_embeddings=self._normalize,
            show_progress_bar=True,
        )

    # ------------------------------------------------------------------ #
    #  Retrieval (READ ONLY — no indexing methods)                         #
    # ------------------------------------------------------------------ #

    def similarity_search(self, query: str, k: int = 5) -> dict:
        """
        Search the collection and return results filtered by
        SIMILARITY_THRESHOLD.  ChromaDB distances are cosine distances,
        so a similarity of 0.7 maps to a maximum allowed distance of 0.3.
        """
        collection = self.client.get_collection(self._collection_name)

        # Query prefix — critical for E5 retrieval quality
        query_embedding = self.create_embeddings([query], is_query=True)[0]

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # Apply threshold filter
        filtered_docs, filtered_meta, filtered_dists = [], [], []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if dist <= self._distance_threshold:
                filtered_docs.append(doc)
                filtered_meta.append(meta)
                filtered_dists.append(dist)
            else:
                print(
                    f"Filtered out chunk (distance {dist:.3f} "
                    f"> threshold {self._distance_threshold:.3f})"
                )

        return {
            "documents": [filtered_docs],
            "metadatas": [filtered_meta],
            "distances": [filtered_dists],
        }
