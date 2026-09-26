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
import hashlib
import logging
from urllib.parse import urlparse

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Reuse the same header/footer-aware loaders used by DocumentProcessor
from src.processing.document_loader import DocumentProcessor

logger = logging.getLogger(__name__)


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
        self.client = self._build_chroma_client()
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
    #  Client construction                                                 #
    # ------------------------------------------------------------------ #

    def _build_chroma_client(self):
        """
        Create the Chroma client.

        ``CHROMA_API_IMPL=rest`` connects to a Chroma server (Cloud Run)
        over HTTP(S); anything else keeps the local persistent store.
        """
        impl = str(getattr(self.config, "CHROMA_API_IMPL", "persistent")).strip().lower()
        if impl in ("rest", "http", "server"):
            host, port, ssl = self._parse_chroma_server()
            logger.info("Chroma client: REST at %s:%s (ssl=%s)", host, port, ssl)
            # Pass the resolved address explicitly: chromadb's own Settings
            # also read CHROMA_SERVER_HOST from the environment, and a
            # mismatch between the two would raise on client creation.
            return chromadb.HttpClient(
                host=host,
                port=port,
                ssl=ssl,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    chroma_server_host=host,
                    chroma_server_http_port=port,
                    chroma_server_ssl_enabled=ssl,
                ),
            )
        logger.info("Chroma client: persistent at %s", self.config.VECTOR_DB_PATH)
        return chromadb.PersistentClient(
            path=self.config.VECTOR_DB_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def _parse_chroma_server(self) -> tuple[str, int, bool]:
        """
        Resolve the Chroma server address from config.

        CHROMA_SERVER_HOST may be a bare host (``chroma.internal``) or a
        full URL (``https://chromadb-service-xxx.a.run.app``), in which
        case scheme and port are taken from the URL.
        """
        raw = str(getattr(self.config, "CHROMA_SERVER_HOST", "localhost")).strip()
        port = int(getattr(self.config, "CHROMA_SERVER_PORT", 8000))
        ssl = bool(getattr(self.config, "CHROMA_SERVER_SSL", False))
        if "://" in raw:
            parsed = urlparse(raw)
            host = parsed.hostname or "localhost"
            ssl = parsed.scheme == "https"
            port = parsed.port or (443 if ssl else 8000)
            return host, port, ssl
        return raw or "localhost", port, ssl

    # ------------------------------------------------------------------ #
    #  Embeddings                                                          #
    # ------------------------------------------------------------------ #

    def create_embeddings(
        self,
        texts: list[str],
        is_query: bool = False,
        show_progress_bar: bool | None = None,
    ) -> np.ndarray:
        """
        Encode texts with the correct E5 prefix.
          is_query=False  →  'passage: '  (index / add time)
          is_query=True   →  'query: '    (search time)

        Progress bars are only shown while indexing; single queries default
        to ``show_progress_bar=False`` to keep the logs quiet.
        """
        prefix = self.QUERY_PREFIX if is_query else self.PASSAGE_PREFIX
        prefixed = [prefix + t for t in texts]
        if show_progress_bar is None:
            show_progress_bar = not is_query
        return self.embeddings_model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
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
            return self.client.get_collection("university_knowledge").count() > 0
        except Exception:
            return False

    def fetch_corpus(self) -> dict:
        """
        Return the entire collection as
        ``{"ids": [...], "documents": [...], "metadatas": [...]}``.

        Used to build the BM25 index; reading does not require the
        embedding model.
        """
        collection = self.client.get_collection("university_knowledge")
        result = collection.get(include=["documents", "metadatas"])
        return {
            "ids": list(result.get("ids") or []),
            "documents": list(result.get("documents") or []),
            "metadatas": list(result.get("metadatas") or []),
        }

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
            logger.warning("No documents supplied")
            return

        # ── Resolve file paths to LangChain Document objects ──────────── #
        if isinstance(documents[0], str):
            processed: list = []
            for file_path in documents:
                if not os.path.isfile(file_path):
                    logger.warning("File not found — %s", file_path)
                    continue
                ext = os.path.splitext(file_path)[1].lower()
                if ext in (".docx", ".pdf"):
                    # DocumentProcessor handles both formats and strips
                    # headers/footers before returning chunks
                    chunks = self._loader.process_file(file_path)
                    processed.extend(chunks)
                else:
                    logger.warning("Unsupported file type '%s' — %s", ext, file_path)

            if not processed:
                logger.warning("No valid documents were loaded from paths")
                return
            documents = processed

        # ── Single, non-duplicated upsert block ───────────────────────── #
        collection = self.client.get_or_create_collection(
            name="university_knowledge",
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
        logger.info("Upserted %d chunks into 'university_knowledge'.", len(texts))

    # ------------------------------------------------------------------ #
    #  Retrieval                                                           #
    # ------------------------------------------------------------------ #

    def similarity_search(self, query: str, k: int = 5) -> dict:
        """
        Search the collection and return results filtered by
        SIMILARITY_THRESHOLD.  ChromaDB distances are cosine distances,
        so a similarity of 0.7 maps to a maximum allowed distance of 0.3.

        The result keeps chunk ids so the hybrid retriever can fuse this
        ranking with the BM25 ranking by a common key.
        """
        collection = self.client.get_collection("university_knowledge")

        # Use the query prefix — this is critical for E5 retrieval quality
        query_embedding = self.create_embeddings(
            [query], is_query=True, show_progress_bar=False
        )[0]

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # Apply threshold filter
        filtered_ids, filtered_docs, filtered_meta, filtered_dists = [], [], [], []
        for chunk_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if dist <= self.distance_threshold:
                filtered_ids.append(chunk_id)
                filtered_docs.append(doc)
                filtered_meta.append(meta)
                filtered_dists.append(dist)
            else:
                logger.debug(
                    "Filtered out chunk (distance %.3f > threshold %.3f)",
                    dist,
                    self.distance_threshold,
                )

        return {
            "ids": [filtered_ids],
            "documents": [filtered_docs],
            "metadatas": [filtered_meta],
            "distances": [filtered_dists],
        }