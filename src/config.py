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
    def _get_env_int(
        cls,
        canonical_key: str,
        legacy_key: Optional[str] = None,
        default_value: int = 0,
    ) -> int:
        """Helper method to get integer values from environment variables with fallback support."""
        value = os.getenv(canonical_key)
        if value is None and legacy_key:
            value = os.getenv(legacy_key)
        if value is None:
            return default_value
        return int(value)

    @classmethod
    def _get_env_str(
        cls,
        canonical_key: str,
        legacy_key: Optional[str] = None,
        default_value: str = "",
    ) -> str:
        """Helper method to get string values from environment variables with fallback support."""
        value = os.getenv(canonical_key)
        if value is None and legacy_key:
            value = os.getenv(legacy_key)
        return value if value is not None else default_value

    @classmethod
    def from_env(cls) -> "Config":
        # Prefer canonical keys but support legacy aliases used in README/.env.example
        repo_url = cls._get_env_str("GIT_REPO_URL", "REPO_URL", cls.repo_url)
        branch = cls._get_env_str("GIT_BRANCH", "BRANCH", cls.branch)
        repo_cache_dir = os.getenv("REPO_CACHE_DIR", cls.repo_cache_dir)

        qdrant_url = os.getenv("QDRANT_URL", cls.qdrant_url)
        qdrant_collection = cls._get_env_str(
            "COLLECTION_NAME", "QDRANT_COLLECTION", cls.qdrant_collection
        )
        vector_size = cls._get_env_int("VECTOR_SIZE", default_value=cls.vector_size)

        ollama_url = os.getenv("OLLAMA_URL", cls.ollama_url)
        embedding_model = cls._get_env_str(
            "OLLAMA_MODEL", "EMBEDDING_MODEL", cls.embedding_model
        )

        max_workers = cls._get_env_int("MAX_WORKERS", default_value=cls.max_workers)
        max_document_chars = cls._get_env_int(
            "MAX_DOCUMENT_CHARS", default_value=cls.max_document_chars
        )
        # Support MAX_TOKEN_LENGTH (README/.env.example) and MAX_TOKENS (internal)
        max_tokens = cls._get_env_int("MAX_TOKEN_LENGTH", "MAX_TOKENS", cls.max_tokens)
        chunk_threshold = cls._get_env_int(
            "CHUNK_THRESHOLD", default_value=cls.chunk_threshold
        )
        chunk_overlap = cls._get_env_int(
            "CHUNK_OVERLAP", default_value=cls.chunk_overlap
        )

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
