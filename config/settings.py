from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # LLM Configuration
    LLM_PROVIDER: str = "ollama"  # ollama or llama_cpp
    LLM_MODEL: str = "llama3.1:latest"
    # True  → local Ollama server (LLM_OLLAMA_BASE_URL)
    # False → remote OpenAI-compatible server (LLM_REMOTE_* below)
    LLM_USE_LOCAL: bool = False
    LLM_TEMPERATURE: float = 0.0
    # Upper bound on generated answers (tokens). Kept low so responses stay
    # concise and generation finishes quickly on CPU.
    LLM_MAX_TOKENS: int = 512
    LLM_NUM_CTX: int = 8192        # Ollama context window (tokens)
    LLM_OLLAMA_BASE_URL: str = "http://localhost:11434"
    # Repetition control — guards against degenerate loops (llama models under
    # greedy decoding sometimes repeat the same phrase endlessly).
    # Local Ollama: repeat_penalty (1.0 = off) and repeat_last_n.
    # Remote endpoint: frequency_penalty = repeat_penalty - 1.
    LLM_REPEAT_PENALTY: float = 1.2
    LLM_REPEAT_LAST_N: int = 256   # local Ollama only

    # Remote OpenAI-compatible LLM (used when LLM_USE_LOCAL is False).
    # BASE_URL is the OpenAI-compatible root; "/chat/completions" is appended.
    # LLM_SERVER_URL is accepted as an alias so deployments can inject the
    # endpoint under a single, generic name.
    LLM_REMOTE_BASE_URL: str = Field(
        default="http://dyres.pmf.kg.ac.rs:11500/v1",
        validation_alias=AliasChoices("LLM_REMOTE_BASE_URL", "LLM_SERVER_URL"),
    )
    LLM_REMOTE_MODEL: str = "llama3.1:8b"
    LLM_REMOTE_API_KEY: str = ""
    LLM_REMOTE_TIMEOUT: float = 300.0  # read timeout (seconds)
    
    # Embeddings
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    EMBEDDING_DEVICE: str = "cpu"  # or "cuda"
    
    # Vector DB
    VECTOR_DB_TYPE: str = "chromadb"  # or "qdrant"
    VECTOR_DB_PATH: str = "./data/vector_store"
    # Chroma client mode:
    #   "persistent" → local on-disk store at VECTOR_DB_PATH
    #   "rest"       → remote Chroma server (Cloud Run) over HTTP(S)
    CHROMA_API_IMPL: str = "persistent"
    CHROMA_SERVER_HOST: str = "localhost"
    CHROMA_SERVER_PORT: int = 8000
    CHROMA_SERVER_SSL: bool = False
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Retrieval
    TOP_K_RESULTS: int = 5
    # Follow-ups asking for additional information about the previous answer
    # retrieve deeper so the model gets new chunks instead of repeating.
    MORE_INFO_TOP_K: int = 10
    SIMILARITY_THRESHOLD: float = 0.7
    RETRIEVAL_CANDIDATES: int = 30      # per-retriever pool size before fusion
    HYBRID_RETRIEVAL_ENABLED: bool = False  # BM25 fusion disabled; dense cosine only
    RRF_K: int = 60                     # Reciprocal Rank Fusion constant
    BM25_FUSION_WEIGHT: float = 0.1     # dense rank weight is always 1.0
    BM25_CACHE_PATH: str = "./data/vector_store/bm25_cache.pkl"

    # Cross-encoder rerank (optional, CPU-friendly multilingual model)
    RERANK_ENABLED: bool = False
    RERANK_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    RERANK_CANDIDATES: int = 20

    # Query rewrite
    REWRITE_TIMEOUT_SECONDS: float = 12.0
    REWRITE_SEED: int = 42           # fixed sampling seed → reproducible rewrites

    # Translation
    DEFAULT_SCRIPT: str = "latinc"  # or "cyrillic"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    RATE_LIMIT: int = 100  # requests per minute
    
    # Redis (optional)
    REDIS_URL: Optional[str] = "redis://localhost:6379"
    
    class Config:
        env_file = ".env"

settings = Settings()