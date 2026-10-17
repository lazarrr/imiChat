from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # `.env` is not versioned and is shared by every branch/deployment, so it
    # regularly carries variables this revision does not know about.  Ignore
    # those instead of refusing to start — pydantic-settings forbids unknown
    # keys by default, which turns a stale `.env` line into a ValidationError
    # at import time.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM Configuration
    LLM_PROVIDER: str = "ollama"  # ollama or llama_cpp
    LLM_MODEL: str = "llama3.1:latest"  # model on the local Ollama server
    # True  → local Ollama server (LLM_OLLAMA_BASE_URL)
    # False → remote OpenAI-compatible server (LLM_REMOTE_* below)
    LLM_USE_LOCAL: bool = False
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 2048
    LLM_NUM_CTX: int = 8192        # Ollama context window (tokens)
    LLM_OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Remote OpenAI-compatible LLM (used when LLM_USE_LOCAL=False).
    # BASE_URL is the API root; "/chat/completions" is appended to it.
    # API_KEY is sent as "Authorization: Bearer <key>" (empty = no header).
    LLM_REMOTE_BASE_URL: str = "http://dyres.pmf.kg.ac.rs:11500/v1"
    LLM_REMOTE_MODEL: str = "llama3.1:8b"
    LLM_REMOTE_API_KEY: str = ""
    LLM_REMOTE_TIMEOUT: float = 300.0  # read timeout (seconds)

    # Repetition control — guards against degenerate loops (llama models under
    # greedy decoding sometimes repeat the same phrase endlessly).
    # Local Ollama: repeat_penalty (1.0 = off) and repeat_last_n.
    # Remote endpoint: frequency_penalty = repeat_penalty - 1.
    LLM_REPEAT_PENALTY: float = 1.2
    LLM_REPEAT_LAST_N: int = 256   # local Ollama only

    
    # Embeddings
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    EMBEDDING_DEVICE: str = "cpu"  # or "cuda"
    
    # Vector DB
    VECTOR_DB_TYPE: str = "chromadb"  # or "qdrant"
    VECTOR_DB_PATH: str = "./data/vector_store"
    CHROMA_COLLECTION: str = "university_knowledge"  # e.g. university_knowledge_test
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Retrieval
    TOP_K_RESULTS: int = 5       # chunks the LLM gets per question
    RETRIEVAL_FETCH_K: int = 20  # candidates fetched before the version policy
    SIMILARITY_THRESHOLD: float = 0.7

    # Better retrieval (optional) — each step is switched on/off on its own;
    # `python run.py eval --compare` scores every combination on the golden
    # questions, so you can see what each step changes.
    QUERY_SYNONYMS: bool = False      # add official terms for student wording (SYNONYMS_FILE)
    SYNONYMS_FILE: str = "config/synonyms.txt"
    HYBRID_SEARCH: bool = False       # fuse BM25 keyword hits (exact names, codes) with vector hits
    RERANKER_ENABLED: bool = False    # cross-encoder re-scores candidates, keeps the best TOP_K_RESULTS
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"  # multilingual, ~2.3 GB download
    RERANKER_DEVICE: str = "cpu"      # "cuda" if a GPU is available
    RERANKER_MIN_SCORE: float = 0.0   # drop chunks the reranker scores below this (0–1); 0 = keep all
    EVAL_QUESTIONS_FILE: str = "../scraper/eval_questions.yaml"  # golden questions for `run.py eval`
    
    # Translation
    DEFAULT_SCRIPT: str = "latinc"  # or "cyrillic"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    RATE_LIMIT: int = 100  # /chat requests per minute per client IP (0 = no limit)

    # Request trace log — one JSON object per answered turn, written by the
    # API to REQUEST_LOG_PATH (see src/api/request_log.py).  It records the
    # whole pipeline: the question as typed, how it was normalized and
    # rewritten, which retrieval strategies were tried, the chunks that were
    # sent to the LLM, and the answer.  A separate visualization project
    # reads this file; the chatbot never reads it back.
    REQUEST_LOG_ENABLED: bool = True
    REQUEST_LOG_PATH: str = "./logs/requests.jsonl"
    REQUEST_LOG_MAX_BYTES: int = 20_000_000  # rotate after ~20 MB
    REQUEST_LOG_BACKUPS: int = 5             # requests.jsonl.1 … .5
    REQUEST_LOG_MAX_TEXT: int = 8000         # per-field cap (chars) for prompt/context/chunk text
    REQUEST_LOG_MAX_EXCLUDED: int = 40       # dropped chunks logged per turn (the count is always logged)

    # Redis (optional) — when set, chat sessions and rate-limit counters live
    # in Redis (shared by all API workers, kept across restarts); otherwise
    # they are kept in the API process's memory.
    REDIS_URL: Optional[str] = None  # e.g. "redis://localhost:6379/0"

settings = Settings()