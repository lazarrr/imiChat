# src/core/vector_store.py
import os
import hashlib
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import numpy as np
import docx2txt
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStoreManager:
    # E5 models require these prefixes — without them, similarity search returns garbage
    QUERY_PREFIX = "query: "
    PASSAGE_PREFIX = "passage: "

    def __init__(self, config):
        self.config = config
        self.embeddings_model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            device=config.EMBEDDING_DEVICE
        )
        self.client = chromadb.PersistentClient(
            path=config.VECTOR_DB_PATH,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        # Cosine distance threshold: similarity 0.7 → distance < 0.3
        self.distance_threshold = 1.0 - config.SIMILARITY_THRESHOLD

    def create_embeddings(self, texts: list, is_query: bool = False) -> np.ndarray:
        """
        Create embeddings with the correct E5 prefix.
        - is_query=True  → 'query: ' prefix    (used at search time)
        - is_query=False → 'passage: ' prefix  (used at index time)
        """
        prefix = self.QUERY_PREFIX if is_query else self.PASSAGE_PREFIX
        prefixed = [prefix + t for t in texts]
        return self.embeddings_model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def _content_id(self, text: str) -> str:
        """Stable, content-derived ID — makes add_documents idempotent across restarts."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _load_docx(self, file_path: str) -> list:
        """Extract text from a .docx file and split into chunks."""
        text = docx2txt.process(file_path)
        docs = self.text_splitter.create_documents(
            [text],
            metadatas=[{"source": file_path, "type": "docx"}]
        )
        return docs

    def is_populated(self) -> bool:
        """Check whether the collection already has documents indexed."""
        try:
            collection = self.client.get_collection("university_knowledge")
            return collection.count() > 0
        except Exception:
            return False

    def add_documents(self, documents: list, metadata: list = None):
        """
        Add documents to the vector store.
        Accepts LangChain Document objects or .docx file paths.
        Safe to call multiple times — uses content-hashed IDs and upsert.
        """
        if not documents:
            print("Warning: No documents to add to vector store")
            return

        # Resolve file paths to LangChain Document objects
        if isinstance(documents[0], str):
            processed_docs = []
            for file_path in documents:
                if not os.path.isfile(file_path):
                    print(f"Warning: File not found: {file_path}")
                    continue
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".docx":
                    docs = self._load_docx(file_path)
                    processed_docs.extend(docs)
                    print(f"Processed DOCX: {os.path.basename(file_path)} — {len(docs)} chunks")
                else:
                    print(f"Warning: Unsupported file type '{ext}' for {file_path}")

            if not processed_docs:
                print("Warning: No valid documents were loaded")
                return
            documents = processed_docs

        # --- Single, non-duplicated add block ---
        collection = self.client.get_or_create_collection(
            name="university_knowledge",
            metadata={"hnsw:space": "cosine"}
        )

        texts = [doc.page_content for doc in documents]

        # BUG FIX 1: use passage prefix for indexing
        embeddings = self.create_embeddings(texts, is_query=False)

        if metadata is None:
            metadata = [doc.metadata for doc in documents]

        # BUG FIX 2: content-hashed IDs → idempotent across restarts
        ids = [self._content_id(t) for t in texts]

        # upsert instead of add → safe to call repeatedly
        collection.upsert(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadata,
            ids=ids
        )
        print(f"Upserted {len(texts)} chunks into 'university_knowledge'.")

    def similarity_search(self, query: str, k: int = 5) -> dict:
        """
        Search for similar documents.
        Results are filtered by SIMILARITY_THRESHOLD from config.
        """
        collection = self.client.get_collection("university_knowledge")

        # BUG FIX 3: use query prefix for search
        query_embedding = self.create_embeddings([query], is_query=True)[0]

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        # BUG FIX 4: apply similarity threshold (cosine distance < 1 - threshold)
        filtered_docs, filtered_meta, filtered_dists = [], [], []
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
                print(f"Filtered out chunk (distance {dist:.3f} > threshold {self.distance_threshold:.3f})")

        return {
            "documents": [filtered_docs],
            "metadatas": [filtered_meta],
            "distances": [filtered_dists],
        }