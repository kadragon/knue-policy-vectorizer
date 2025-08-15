import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Git repository settings
    repo_url: str = "https://github.com/kadragon/KNUE-Policy-Hub.git"
    branch: str = "main"
    repo_cache_dir: str = "./repo_cache"
    
    # Qdrant settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "knue-policy-idx"
    vector_size: int = 1024
    
    # Ollama settings
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "bge-m3"
    
    # Processing settings
    max_workers: int = 4
    max_document_chars: int = 30000
    max_tokens: int = 8192  # Maximum tokens for embedding service (bge-m3 limit)
    chunk_threshold: int = 800  # Chunking threshold for better semantic search
    chunk_overlap: int = 200  # Overlap tokens between chunks for context continuity
    
    # Logging settings
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "Config":
        # Prefer canonical keys but support legacy aliases used in README/.env.example
        repo_url = os.getenv("GIT_REPO_URL") or os.getenv("REPO_URL", cls.repo_url)
        branch = os.getenv("GIT_BRANCH") or os.getenv("BRANCH", cls.branch)
        repo_cache_dir = os.getenv("REPO_CACHE_DIR", cls.repo_cache_dir)

        qdrant_url = os.getenv("QDRANT_URL", cls.qdrant_url)
        qdrant_collection = (
            os.getenv("COLLECTION_NAME")
            or os.getenv("QDRANT_COLLECTION", cls.qdrant_collection)
        )
        vector_size = int(os.getenv("VECTOR_SIZE", str(cls.vector_size)))

        ollama_url = os.getenv("OLLAMA_URL", cls.ollama_url)
        embedding_model = os.getenv("OLLAMA_MODEL") or os.getenv(
            "EMBEDDING_MODEL", cls.embedding_model
        )

        max_workers = int(os.getenv("MAX_WORKERS", str(cls.max_workers)))
        max_document_chars = int(
            os.getenv("MAX_DOCUMENT_CHARS", str(cls.max_document_chars))
        )
        # Support MAX_TOKEN_LENGTH (README/.env.example) and MAX_TOKENS (internal)
        max_tokens = int(
            os.getenv("MAX_TOKEN_LENGTH")
            or os.getenv("MAX_TOKENS", str(cls.max_tokens))
        )
        chunk_threshold = int(
            os.getenv("CHUNK_THRESHOLD", str(cls.chunk_threshold))
        )
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", str(cls.chunk_overlap)))

        log_level = os.getenv("LOG_LEVEL", cls.log_level)

        return cls(
            repo_url=repo_url,
            branch=branch,
            repo_cache_dir=repo_cache_dir,
            qdrant_url=qdrant_url,
            qdrant_collection=qdrant_collection,
            vector_size=vector_size,
            ollama_url=ollama_url,
            embedding_model=embedding_model,
            max_workers=max_workers,
            max_document_chars=max_document_chars,
            max_tokens=max_tokens,
            chunk_threshold=chunk_threshold,
            chunk_overlap=chunk_overlap,
            log_level=log_level,
        )
