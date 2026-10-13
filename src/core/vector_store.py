"""
vector_store.py
~~~~~~~~~~~~~~~
ChromaDB-backed vector store using intfloat/multilingual-e5-large.

Key fixes applied here (see previous review for full context):
  1. E5 query / passage prefixes applied correctly.
  2. Duplicate add() block removed.
  3. Content-hashed IDs + upsert() → idempotent across restarts.
  4. SIMILARITY_THRESHOLD actually applied to search results.
  5. DOCX loading replaced: docx2txt → python-docx (strips headers/footers).
  6. PDF loading added via pdfplumber (strips headers/footers).
"""

import os
import re
import hashlib

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Reuse the same header/footer-aware loaders used by DocumentProcessor
from src.processing.document_loader import DocumentProcessor

# A question asks about an OLD program version if it names a year
# 2000–2018 or says "stari program", "stara akreditacija", …
_LEGACY_QUERY_RE = re.compile(
    r"\b20(0\d|1[0-8])\b"
    r"|\bstar(i|og|om|im|ih|e|a|u)\s+(studijsk\w+\s+)?(program\w*|akreditacij\w*|plan\w*)"
    r"|\bprethodn\w+\s+akreditacij\w*",
    re.IGNORECASE,
)

# Applied inside the Chroma query (legacy program tables are about half of
# all chunks and would crowd current content out of the candidates).
# $ne also keeps chunks that do not have the key at all.
_CURRENT_ONLY_WHERE = {"$and": [{"status": {"$ne": "legacy"}}, {"outdated": {"$ne": True}}]}


class VectorStoreManager:
    # E5 models require explicit prefixes — omitting them degrades retrieval quality
    QUERY_PREFIX   = "query: "
    PASSAGE_PREFIX = "passage: "

    def __init__(self, config):
        self.config = config
        self.embeddings_model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            device=config.EMBEDDING_DEVICE,
        )
        self.client = chromadb.PersistentClient(
            path=config.VECTOR_DB_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection_name = getattr(config, "CHROMA_COLLECTION", "university_knowledge")
        self.top_k   = getattr(config, "TOP_K_RESULTS", 5)
        self.fetch_k = getattr(config, "RETRIEVAL_FETCH_K", 20)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        # Cosine distance threshold derived from similarity threshold in config.
        # ChromaDB stores cosine distance (0 = identical, 2 = opposite).
        # distance < (1 - similarity_threshold) keeps high-quality results.
        self.distance_threshold = 1.0 - config.SIMILARITY_THRESHOLD

        # Internal loader used when add_documents() receives file paths
        self._loader = DocumentProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )

    # ------------------------------------------------------------------ #
    #  Embeddings                                                          #
    # ------------------------------------------------------------------ #

    def create_embeddings(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """
        Encode texts with the correct E5 prefix.
          is_query=False  →  'passage: '  (index / add time)
          is_query=True   →  'query: '    (search time)
        """
        prefix = self.QUERY_PREFIX if is_query else self.PASSAGE_PREFIX
        prefixed = [prefix + t for t in texts]
        return self.embeddings_model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _content_id(text: str) -> str:
        """Stable, content-derived document ID — makes upsert idempotent."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def is_populated(self) -> bool:
        """True if the collection already has at least one document."""
        try:
            return self.client.get_collection(self.collection_name).count() > 0
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    #  Indexing                                                            #
    # ------------------------------------------------------------------ #

    def add_documents(self, documents: list, metadata: list = None):
        """
        Add documents to the vector store.

        Accepts:
          • LangChain Document objects (.page_content / .metadata)
          • File paths (str) — .docx or .pdf, with headers/footers stripped
        """
        if not documents:
            print("Warning: no documents supplied")
            return

        # ── Resolve file paths to LangChain Document objects ──────────── #
        if isinstance(documents[0], str):
            processed: list = []
            for file_path in documents:
                if not os.path.isfile(file_path):
                    print(f"Warning: file not found — {file_path}")
                    continue
                ext = os.path.splitext(file_path)[1].lower()
                if ext in (".docx", ".pdf"):
                    # DocumentProcessor handles both formats and strips
                    # headers/footers before returning chunks
                    chunks = self._loader.process_file(file_path)
                    processed.extend(chunks)
                else:
                    print(f"Warning: unsupported file type '{ext}' — {file_path}")

            if not processed:
                print("Warning: no valid documents were loaded from paths")
                return
            documents = processed

        # ── Single, non-duplicated upsert block ───────────────────────── #
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        texts      = [doc.page_content for doc in documents]
        embeddings = self.create_embeddings(texts, is_query=False)  # passage prefix

        if metadata is None:
            metadata = [doc.metadata for doc in documents]

        ids = [self._content_id(t) for t in texts]

        # upsert is safe to call repeatedly with the same content
        collection.upsert(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadata,
            ids=ids,
        )
        print(f"Upserted {len(texts)} chunks into '{self.collection_name}'.")

    # ------------------------------------------------------------------ #
    #  Retrieval                                                           #
    # ------------------------------------------------------------------ #

    def similarity_search(self, query: str, k: int | None = None) -> dict:
        """
        Search the collection and return results filtered by
        SIMILARITY_THRESHOLD.  ChromaDB distances are cosine distances,
        so a similarity of 0.7 maps to a maximum allowed distance of 0.3.

        RETRIEVAL_FETCH_K candidates are fetched so that, after legacy
        program versions and duplicates are removed by the version
        policy, k useful chunks remain.  k defaults to TOP_K_RESULTS.
        """
        k = k or self.top_k
        collection = self.client.get_collection(self.collection_name)

        # Use the query prefix — this is critical for E5 retrieval quality
        query_embedding = self.create_embeddings([query], is_query=True)[0]

        wants_old = bool(_LEGACY_QUERY_RE.search(query or ""))
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=max(k, self.fetch_k),
            where=None if wants_old else _CURRENT_ONLY_WHERE,
            include=["documents", "metadatas", "distances"],
        )

        # Apply threshold filter
        filtered_docs, filtered_meta, filtered_dists = [], [], []
        n_filtered = 0
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if dist <= self.distance_threshold:
                filtered_docs.append(doc)
                filtered_meta.append(meta)
                filtered_dists.append(dist)
            else:
                n_filtered += 1
        if n_filtered:
            print(
                f"Filtered out {n_filtered} chunk(s) "
                f"(distance > threshold {self.distance_threshold:.3f})"
            )

        filtered_docs, filtered_meta, filtered_dists = self._apply_version_policy(
            query, filtered_docs, filtered_meta, filtered_dists, k
        )

        return {
            "documents": [filtered_docs],
            "metadatas": [filtered_meta],
            "distances": [filtered_dists],
        }

    @staticmethod
    def _apply_version_policy(
        query: str,
        documents: list,
        metadatas: list,
        distances: list,
        k: int,
    ) -> tuple[list, list, list]:
        """
        Version-aware result policy (same rules as the scraper's
        src/core/retrieval.py, so `run.py search` there shows what the
        chatbot gets):
          1. drop chunks of legacy program versions (metadata
             status == "legacy") and chunks the scraper's DeepSeek check
             flagged as outdated (outdated == True, e.g. an FAQ still
             saying "3 smera"), unless the question asks for an old
             program — this is what stops "2 vs 3 modules" answers
          2. collapse chunks sharing a dedup_key (same text, same version)
          3. keep the best k
        Chunks without these metadata fields pass through unchanged.
        """
        keep_legacy = bool(_LEGACY_QUERY_RE.search(query or ""))
        docs, metas, dists = [], [], []
        seen: set[str] = set()
        for doc, meta, dist in zip(documents, metadatas, distances):
            meta = meta or {}
            if not keep_legacy and (meta.get("status") == "legacy" or meta.get("outdated") is True):
                continue
            key = meta.get("dedup_key")
            if key:
                if key in seen:
                    continue
                seen.add(key)
            docs.append(doc)
            metas.append(meta)
            dists.append(dist)
            if len(docs) >= k:
                break
        return docs, metas, dists