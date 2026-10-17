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

Optional retrieval steps — synonym expansion, hybrid (BM25 + vector)
search and cross-encoder reranking — are switched on/off in config; see
similarity_search().
"""

import os
import re
import hashlib
import logging
import threading
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Reuse the same header/footer-aware loaders used by DocumentProcessor
from src.processing.document_loader import DocumentProcessor
from src.core.keyword_search import BM25Index, SynonymExpander, reciprocal_rank_fusion
from src.core.reranker import Reranker

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


def _is_current(meta: dict) -> bool:
    """_CURRENT_ONLY_WHERE for chunks found outside Chroma (keyword search)."""
    return meta.get("status") != "legacy" and meta.get("outdated") is not True


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

        # Optional retrieval steps (see similarity_search).  The flags can be
        # flipped at runtime — `run.py eval --compare` does — and each
        # component is loaded the first time its step runs.
        self.use_synonyms = getattr(config, "QUERY_SYNONYMS", False)
        self.use_hybrid   = getattr(config, "HYBRID_SEARCH", False)
        self.use_reranker = getattr(config, "RERANKER_ENABLED", False)
        self.rerank_min_score = getattr(config, "RERANKER_MIN_SCORE", 0.0)
        self._synonyms: SynonymExpander | None = None
        self._keyword_index: BM25Index | None = None
        self._reranker: Reranker | None = None
        # One search at a time: the API searches from worker threads (see
        # UniversityQASystem.query_stream), and neither the lazily built
        # components above nor the Hugging Face tokenizers are thread-safe.
        # Answers still stream in parallel; only retrieval takes turns.
        self._search_lock = threading.Lock()
        if self.use_synonyms:
            self._get_synonyms()
        if self.use_reranker:
            self._get_reranker()  # load the model now, not on the first question

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
            show_progress_bar=not is_query,  # no bar for every question
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

    def similarity_search(
        self, query: str, k: int | None = None, diagnostics: bool = False
    ) -> dict:
        """
        Search the collection and return the best k chunks (k defaults to
        TOP_K_RESULTS).

        Always: RETRIEVAL_FETCH_K vector-search candidates filtered by
        SIMILARITY_THRESHOLD (ChromaDB distances are cosine distances, so a
        similarity of 0.7 maps to a maximum allowed distance of 0.3), then
        the version policy (legacy program versions and duplicates
        removed) keeps the best k.

        Optional steps, each switched on in config:
          QUERY_SYNONYMS    search with official terms added to the query
          HYBRID_SEARCH     fuse BM25 keyword hits with the vector hits
                            (reciprocal rank fusion)
          RERANKER_ENABLED  the version policy keeps RETRIEVAL_FETCH_K
                            candidates; a cross-encoder scores them against
                            the question and the best k are kept (none
                            below RERANKER_MIN_SCORE)
        With every step off, this is exactly the plain vector search.
        Searches from several threads run one at a time (_search_lock).

        ``diagnostics=True`` additionally reports the chunks that were
        retrieved but did **not** make it into the result, with the reason
        each one was dropped (``results["excluded"]``, ``results["excluded_total"]``).
        It costs one extra Chroma query without the current-version filter,
        so it is off by default and only the request log turns it on.  The
        visible result is identical either way.
        """
        with self._search_lock:
            return self._search(query, k or self.top_k, diagnostics=diagnostics)

    def _search(self, query: str, k: int, diagnostics: bool = False) -> dict:
        """similarity_search() itself, run while holding _search_lock."""
        collection = self.client.get_collection(self.collection_name)
        wants_old = bool(_LEGACY_QUERY_RE.search(query or ""))
        # Chunks dropped on the way to the final k: reason, text, metadata,
        # distance.  Only filled in when diagnostics are requested.
        excluded: list[dict] = []

        def drop(reason: str, chunk_id, doc, meta, dist, **extra) -> None:
            if not diagnostics:
                return
            excluded.append({
                "chunk_id": chunk_id,
                "reason": reason,
                "distance": dist,
                "metadata": dict(meta or {}),
                "text": doc or "",
                **extra,
            })

        search_query = query
        if self.use_synonyms:
            search_query = self._get_synonyms().expand(query)
            if search_query != query:
                logger.info("Synonyms: searching for %r", search_query)

        # Use the query prefix — this is critical for E5 retrieval quality
        query_embedding = self.create_embeddings([search_query], is_query=True)[0]

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=max(k, self.fetch_k),
            where=None if wants_old else _CURRENT_ONLY_WHERE,
            include=["documents", "metadatas", "distances"],
        )

        # Re-query without the current-version filter, to see what that filter
        # hid: chunks with status="legacy" / outdated=true never appear in
        # `results`, so they can only be reported from a second query.
        if diagnostics and not wants_old:
            try:
                wide = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=min(max(k, self.fetch_k) * 3, 60),
                    where=None,
                    include=["documents", "metadatas", "distances"],
                )
                visible = set(results["ids"][0])
                for chunk_id, doc, meta, dist in zip(
                    wide["ids"][0], wide["documents"][0],
                    wide["metadatas"][0], wide["distances"][0],
                ):
                    if chunk_id in visible:
                        continue
                    reason = ("excluded_outdated" if (meta or {}).get("outdated") is True
                              else "excluded_legacy_version")
                    drop(reason, chunk_id, doc, meta, dist)
            except Exception as e:  # pragma: no cover - diagnostics must not break search
                logger.warning("Diagnostic query failed (%s) - no excluded chunks reported", e)

        # Apply threshold filter.  candidates: chunk id → (doc, meta, distance)
        candidates: dict[str, tuple] = {}
        n_filtered = 0
        for chunk_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if dist <= self.distance_threshold:
                candidates[chunk_id] = (doc, meta, dist)
            else:
                n_filtered += 1
                drop("excluded_threshold", chunk_id, doc, meta, dist,
                     threshold=self.distance_threshold)
        if n_filtered:
            logger.info(
                "Filtered out %d chunk(s) (distance > threshold %.3f)",
                n_filtered, self.distance_threshold,
            )
        ranking = list(candidates)

        if self.use_hybrid:
            hits = self._get_keyword_index(collection).search(
                search_query, self.fetch_k, allow=None if wants_old else _is_current
            )
            for chunk_id, doc, meta in hits:
                candidates.setdefault(chunk_id, (doc, meta, None))  # keyword-only: no distance
            logger.info("Hybrid search: %d vector + %d keyword candidate(s)", len(ranking), len(hits))
            ranking = reciprocal_rank_fusion(ranking, [chunk_id for chunk_id, _, _ in hits])

        filtered_docs, filtered_meta, filtered_dists = self._apply_version_policy(
            query,
            [candidates[c][0] for c in ranking],
            [candidates[c][1] for c in ranking],
            [candidates[c][2] for c in ranking],
            max(k, self.fetch_k) if self.use_reranker else k,
            exclude=lambda reason, doc, meta, dist: drop(reason, None, doc, meta, dist),
        )

        if self.use_reranker and filtered_docs:
            scores = self._get_reranker().score(query, filtered_docs)
            # keep_best(), unrolled, so the chunks the reranker drops are
            # reported instead of silently vanishing.
            pairs = sorted(
                zip(scores, zip(filtered_docs, filtered_meta, filtered_dists)),
                key=lambda pair: pair[0], reverse=True,
            )
            kept, dropped = [], []
            for score, (doc, meta, dist) in pairs:
                if len(kept) < k and score >= self.rerank_min_score:
                    kept.append((score, (doc, meta, dist)))
                else:
                    dropped.append((score, (doc, meta, dist)))
            for score, (doc, meta, dist) in dropped:
                drop("excluded_reranker", None, doc, meta, dist,
                     rerank_score=round(float(score), 4),
                     rerank_min_score=self.rerank_min_score)
            logger.info(
                "Reranker: kept %d of %d candidate(s); scores %s",
                len(kept), len(filtered_docs), [round(score, 3) for score, _ in kept],
            )
            filtered_docs  = [doc for _, (doc, _, _) in kept]
            filtered_meta  = [meta for _, (_, meta, _) in kept]
            filtered_dists = [dist for _, (_, _, dist) in kept]

        payload = {
            "documents": [filtered_docs],
            "metadatas": [filtered_meta],
            "distances": [filtered_dists],
        }
        if diagnostics:
            payload["excluded"] = excluded[:100]
            payload["excluded_total"] = len(excluded)
        return payload

    @staticmethod
    def _apply_version_policy(
        query: str,
        documents: list,
        metadatas: list,
        distances: list,
        k: int,
        exclude=None,
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

        ``exclude(reason, doc, meta, dist)`` is called for every dropped
        chunk, which is how the request log explains why a candidate did not
        reach the model.
        """
        keep_legacy = bool(_LEGACY_QUERY_RE.search(query or ""))
        docs, metas, dists = [], [], []
        seen: set[str] = set()
        for index, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            meta = meta or {}
            if not keep_legacy and (meta.get("status") == "legacy" or meta.get("outdated") is True):
                if exclude:
                    reason = "excluded_outdated" if meta.get("outdated") is True \
                        else "excluded_legacy_version"
                    exclude(reason, doc, meta, dist)
                continue
            key = meta.get("dedup_key")
            if key:
                if key in seen:
                    if exclude:
                        exclude("excluded_duplicate", doc, meta, dist)
                    continue
                seen.add(key)
            docs.append(doc)
            metas.append(meta)
            dists.append(dist)
            if len(docs) >= k:
                break
        return docs, metas, dists

    # ------------------------------------------------------------------ #
    #  Optional retrieval steps — components loaded on first use          #
    # ------------------------------------------------------------------ #

    def _get_synonyms(self) -> SynonymExpander:
        if self._synonyms is None:
            path = Path(getattr(self.config, "SYNONYMS_FILE", "config/synonyms.txt"))
            if not path.is_absolute():
                path = _PROJECT_ROOT / path
            if path.is_file():
                self._synonyms = SynonymExpander.from_file(path)
                logger.info("Synonyms: %d group(s) from %s", len(self._synonyms.groups), path)
            else:
                logger.warning("Synonyms file not found — %s; searching without synonyms", path)
                self._synonyms = SynonymExpander([])
        return self._synonyms

    def _get_keyword_index(self, collection) -> BM25Index:
        """BM25 index over the whole collection, rebuilt when its size changes
        (e.g. after the scraper ingested or removed pages)."""
        count = collection.count()
        if self._keyword_index is None or len(self._keyword_index) != count:
            data = collection.get(include=["documents", "metadatas"])
            self._keyword_index = BM25Index(data["ids"], data["documents"], data["metadatas"])
            logger.info("Keyword index built over %d chunks", count)
        return self._keyword_index

    def _get_reranker(self) -> Reranker:
        if self._reranker is None:
            model = getattr(self.config, "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
            device = getattr(self.config, "RERANKER_DEVICE", "cpu")
            logger.info("Loading reranker %s on %s...", model, device)
            self._reranker = Reranker(model, device=device)
        return self._reranker