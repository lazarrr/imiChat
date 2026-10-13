from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # LLM Configuration
    LLM_PROVIDER: str = "ollama"  # ollama or llama_cpp
    LLM_MODEL: str = "llama3.1:latest"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 2048
    LLM_NUM_CTX: int = 8192        # Ollama context window (tokens)
    LLM_OLLAMA_BASE_URL: str = "http://localhost:11434"
    
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
    
    # Translation
    DEFAULT_SCRIPT: str = "latinc"  # or "cyrillic"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    RATE_LIMIT: int = 100  # /chat requests per minute per client IP (0 = no limit)

    # Redis (optional) — when set, chat sessions and rate-limit counters live
    # in Redis (shared by all API workers, kept across restarts); otherwise
    # they are kept in the API process's memory.
    REDIS_URL: Optional[str] = None  # e.g. "redis://localhost:6379/0"
    
    class Config:
        env_file = ".env"

settings = Settings()