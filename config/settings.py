from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # LLM Configuration
    LLM_PROVIDER: str = "ollama"  # ollama or llama_cpp
    LLM_MODEL: str = "llama3.1:latest"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 2048
    
    # Embeddings
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    EMBEDDING_DEVICE: str = "cpu"  # or "cuda"
    
    # Vector DB
    VECTOR_DB_TYPE: str = "chromadb"  # or "qdrant"
    VECTOR_DB_PATH: str = "./data/vector_store"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Retrieval
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Translation
    DEFAULT_SCRIPT: str = "latinc"  # or "cyrillic"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    RATE_LIMIT: int = 100  # requests per minute
    
    # Redis (optional)
    REDIS_URL: Optional[str] = "redis://localhost:6379"

    # Crawler
    CRAWLER_MAX_DEPTH: int = 1
    CRAWLER_DELAY: float = 1.0
    CRAWLER_TIMEOUT: int = 15
    CRAWLER_USER_AGENT: str = "Mozilla/5.0 (compatible; UniversityRAGBot/1.0)"

    # Crawl history — persists which URLs were already ingested so
    # subsequent runs don't re-crawl the same pages.
    # Set to False to force a full re-crawl on every run.
    CRAWL_HISTORY_ENABLED: bool = True
    CRAWL_HISTORY_PATH: str = "./data/crawl_history.json"

    # Crawler URL filtering
    #   "include" — only crawl URLs whose path matches CRAWLER_INCLUDE_PATTERNS
    #   "exclude" — skip URLs whose path matches CRAWLER_EXCLUDE_PATTERNS
    #   "none"    — crawl everything (no filtering)
    CRAWLER_FILTER_MODE: str = "include"

    # TEST MODE — study programs + bulletin board only.
    # Add more patterns below when you want to expand. Examples are commented out.
    CRAWLER_INCLUDE_PATTERNS: list[str] = [
        r"^/$",                        # homepage (needed to discover links)
        r"^/studijski-programi",       # study programs overview
        r"^/matematika-studije",       # math studies + subpages
        r"^/informatika-studije",      # informatics studies + subpages
        r"^/oglasna-tabla",            # bulletin board
        r""  # homepage (for absolute URLs)
        # r"^/upis-studije",           # enrollment info
        # r"^/o-institutu",            # about the institute
        # r"^/nastavno-osoblje",       # teaching staff
        # r"^/brucosi",                # freshmen info
        # r"^/rasporedi",              # class schedule
        # r"^/raspored-ispita",        # exam schedule
        # r"^/strucna-praksa",         # professional practice
        # r"^/uslovni-predmeti",       # conditional courses
        # r"^/ppm-grupa-predmeta",     # PPM course group
        # r"^/zavrsni-master",         # master theses
        # r"^/projekti-",              # research projects
        # r"^/publikacije-",           # publications
        # r"^/seminar-scimi",          # ScIMI seminar
        # r"^/kontakt",                # contact page
        # r"^/vesti/",                 # news articles
    ]

    # Used only when CRAWLER_FILTER_MODE == "exclude"
    CRAWLER_EXCLUDE_PATTERNS: list[str] = [
        r"blog\.imi\.pmf\.kg\.ac\.rs",
        r"/images/",
        r"/office365",
    ]
    
    class Config:
        env_file = ".env"

settings = Settings()