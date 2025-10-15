# CONFIG GROUP - Configuration management and settings
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

from src.utils.providers import EmbeddingProvider, ProviderConfig, VectorProvider

load_dotenv()


@dataclass
class Config:
    # Git repository settings
    repo_url: str = "https://github.com/kadragon/KNUE-Policy-Hub.git"
    branch: str = "main"
    repo_cache_dir: str = "./repo_cache"

    # Provider selection
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    vector_provider: VectorProvider = VectorProvider.QDRANT_CLOUD

    # Qdrant settings (cloud)
    qdrant_collection: str = "knue_policies"
    vector_size: int = 1536

    # Qdrant collection for KNUE web boards
    qdrant_board_collection: str = "www-board-data"

    # Qdrant Cloud settings
    qdrant_cloud_url: str = ""
    qdrant_api_key: str = ""
    qdrant_cluster_region: str = ""

    # Cloudflare R2 settings
    cloudflare_account_id: str = ""
    cloudflare_r2_access_key_id: str = ""
    cloudflare_r2_secret_access_key: str = ""
    cloudflare_r2_bucket: str = ""
    cloudflare_r2_endpoint: str = ""
    cloudflare_r2_key_prefix: str = ""
    cloudflare_r2_soft_delete_enabled: bool = False
    cloudflare_r2_soft_delete_prefix: str = "deleted/"

    # OpenAI settings
    openai_api_key: str = ""
    openai_model: str = "text-embedding-3-large"
    openai_base_url: str = "https://api.openai.com/v1"

    # Processing settings
    max_workers: int = 4
    max_document_chars: int = 30000
    max_tokens: int = 8192  # Maximum tokens for embedding service
    chunk_threshold: int = 800  # Chunking threshold for better semantic search
    chunk_overlap: int = 200  # Overlap tokens between chunks for context continuity

    # KNUE Board ingest settings
    board_rss_template: str = "https://www.knue.ac.kr/rssBbsNtt.do?bbsNo={board_idx}"
    board_indices: Tuple[int, ...] = (
        25,
        26,
        11,
        207,
        28,
        256,
    )
    board_chunk_size: int = 800
    board_chunk_overlap: int = 200
    board_max_age_days: int = 1  # Only ingest posts within this many days
    # Retention: delete board items older than this many days (default 730 ~ 2 years)
    board_retention_days: int = 730
    # Board-specific title prefix skip map: {board_idx: ["[교육봉사]", ...]}
    board_skip_prefix_map: Dict[int, Tuple[str, ...]] = field(default_factory=dict)
    # Embedding batch size when processing board items
    board_embed_batch_size: int = 32
    # Adaptive batching + retry/backoff
    board_embed_retry_max: int = 3
    board_embed_backoff_base: float = 0.5

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
    def _get_env_float(
        cls,
        key: str,
        default_value: float,
    ) -> float:
        """Helper method to get float values from environment variables."""
        v = os.getenv(key)
        if v is None:
            return default_value
        try:
            return float(v)
        except ValueError:
            return default_value

    @classmethod
    def from_env(cls) -> "Config":
        # Prefer canonical keys but support legacy aliases used in README/.env.example
        repo_url = cls._get_env_str("GIT_REPO_URL", "REPO_URL", cls.repo_url)
        branch = cls._get_env_str("GIT_BRANCH", "BRANCH", cls.branch)
        repo_cache_dir = os.getenv("REPO_CACHE_DIR", cls.repo_cache_dir)

        # Provider selection
        # Prefer explicit provider envs; otherwise infer sensible defaults from legacy-specific vars
        embedding_provider_str = os.getenv("EMBEDDING_PROVIDER")
        if not embedding_provider_str:
            if (
                os.getenv("OPENAI_API_KEY")
                or os.getenv("OPENAI_MODEL")
                or os.getenv("OPENAI_BASE_URL")
            ):
                embedding_provider_str = "openai"
            else:
                embedding_provider_str = "openai"

        vector_provider_str = os.getenv("VECTOR_PROVIDER")
        if not vector_provider_str:
            if os.getenv("QDRANT_CLOUD_URL") or os.getenv("QDRANT_API_KEY"):
                vector_provider_str = "qdrant_cloud"
            else:
                vector_provider_str = "qdrant_cloud"

        try:
            embedding_provider = EmbeddingProvider(embedding_provider_str)
        except ValueError:
            raise ValueError(f"Invalid embedding provider: {embedding_provider_str}")

        try:
            vector_provider = VectorProvider(vector_provider_str)
        except ValueError:
            raise ValueError(f"Invalid vector provider: {vector_provider_str}")

        qdrant_collection = cls._get_env_str(
            "COLLECTION_NAME", "QDRANT_COLLECTION", cls.qdrant_collection
        )
        vector_size = cls._get_env_int("VECTOR_SIZE", default_value=cls.vector_size)

        # Qdrant Cloud settings
        qdrant_cloud_url = os.getenv("QDRANT_CLOUD_URL", cls.qdrant_cloud_url)
        qdrant_api_key = os.getenv("QDRANT_API_KEY", cls.qdrant_api_key)
        qdrant_cluster_region = os.getenv(
            "QDRANT_CLUSTER_REGION", cls.qdrant_cluster_region
        )

        # OpenAI settings
        openai_api_key = os.getenv("OPENAI_API_KEY", cls.openai_api_key)
        openai_model = os.getenv("OPENAI_MODEL", cls.openai_model)
        openai_base_url = os.getenv("OPENAI_BASE_URL", cls.openai_base_url)

        # Cloudflare R2 settings
        cloudflare_account_id = os.getenv(
            "CLOUDFLARE_ACCOUNT_ID", cls.cloudflare_account_id
        )
        cloudflare_r2_access_key_id = os.getenv(
            "CLOUDFLARE_R2_ACCESS_KEY_ID", cls.cloudflare_r2_access_key_id
        )
        cloudflare_r2_secret_access_key = os.getenv(
            "CLOUDFLARE_R2_SECRET_ACCESS_KEY", cls.cloudflare_r2_secret_access_key
        )
        cloudflare_r2_bucket = os.getenv(
            "CLOUDFLARE_R2_BUCKET", cls.cloudflare_r2_bucket
        )
        cloudflare_r2_endpoint = os.getenv(
            "CLOUDFLARE_R2_ENDPOINT", cls.cloudflare_r2_endpoint
        )
        cloudflare_r2_key_prefix = os.getenv(
            "CLOUDFLARE_R2_KEY_PREFIX", cls.cloudflare_r2_key_prefix
        )
        cloudflare_r2_soft_delete_prefix = os.getenv(
            "CLOUDFLARE_R2_SOFT_DELETE_PREFIX", cls.cloudflare_r2_soft_delete_prefix
        )
        cloudflare_r2_soft_delete_env = os.getenv(
            "CLOUDFLARE_R2_SOFT_DELETE_ENABLED", "false"
        )
        cloudflare_r2_soft_delete_enabled = str(
            cloudflare_r2_soft_delete_env
        ).lower() in {"1", "true", "yes", "on"}

        # Processing settings
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

        # KNUE Board ingest settings
        qdrant_board_collection = os.getenv(
            "QDRANT_BOARD_COLLECTION", cls.qdrant_board_collection
        )
        board_rss_template = os.getenv("BOARD_RSS_TEMPLATE", cls.board_rss_template)
        board_indices_raw = os.getenv("BOARD_INDICES")
        if board_indices_raw:
            try:
                board_indices = tuple(
                    int(x.strip()) for x in board_indices_raw.split(",") if x.strip()
                )
            except ValueError:
                raise ValueError(
                    "Invalid BOARD_INDICES. Expected comma-separated integers."
                )
        else:
            board_indices = cls.board_indices

        board_chunk_size = cls._get_env_int(
            "BOARD_CHUNK_SIZE", default_value=cls.board_chunk_size
        )
        board_chunk_overlap = cls._get_env_int(
            "BOARD_CHUNK_OVERLAP", default_value=cls.board_chunk_overlap
        )
        board_max_age_days = cls._get_env_int(
            "BOARD_MAX_AGE_DAYS", default_value=cls.board_max_age_days
        )
        board_retention_days = cls._get_env_int(
            "BOARD_RETENTION_DAYS", default_value=cls.board_retention_days
        )
        # Parse board skip prefix map from JSON env var
        board_skip_prefix_map_env = os.getenv("BOARD_SKIP_PREFIX_MAP", "")
        board_skip_prefix_map: Dict[int, Tuple[str, ...]] = {}
        if board_skip_prefix_map_env:
            try:
                raw = json.loads(board_skip_prefix_map_env)
                if not isinstance(raw, dict):
                    raise ValueError("BOARD_SKIP_PREFIX_MAP must be a JSON object")
                for k, v in raw.items():
                    # Keys may be strings; coerce to int
                    try:
                        key_int = int(k)
                    except Exception:
                        raise ValueError(
                            f"BOARD_SKIP_PREFIX_MAP has non-integer key: {k!r}"
                        )
                    if not isinstance(v, (list, tuple)) or not all(
                        isinstance(s, str) for s in v
                    ):
                        raise ValueError(
                            "BOARD_SKIP_PREFIX_MAP values must be lists of strings"
                        )
                    board_skip_prefix_map[key_int] = tuple(str(s) for s in v)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid BOARD_SKIP_PREFIX_MAP JSON: {e.msg}") from e
        board_embed_batch_size = cls._get_env_int(
            "BOARD_EMBED_BATCH_SIZE", default_value=cls.board_embed_batch_size
        )
        board_embed_retry_max = cls._get_env_int(
            "BOARD_EMBED_RETRY_MAX", default_value=cls.board_embed_retry_max
        )
        board_embed_backoff_base = cls._get_env_float(
            "BOARD_EMBED_BACKOFF_BASE", cls.board_embed_backoff_base
        )

        return cls(
            repo_url=repo_url,
            branch=branch,
            repo_cache_dir=repo_cache_dir,
            embedding_provider=embedding_provider,
            vector_provider=vector_provider,
            qdrant_collection=qdrant_collection,
            vector_size=vector_size,
            qdrant_board_collection=qdrant_board_collection,
            qdrant_cloud_url=qdrant_cloud_url,
            qdrant_api_key=qdrant_api_key,
            qdrant_cluster_region=qdrant_cluster_region,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            openai_base_url=openai_base_url,
            max_workers=max_workers,
            max_document_chars=max_document_chars,
            max_tokens=max_tokens,
            chunk_threshold=chunk_threshold,
            chunk_overlap=chunk_overlap,
            board_rss_template=board_rss_template,
            board_indices=board_indices,
            board_chunk_size=board_chunk_size,
            board_chunk_overlap=board_chunk_overlap,
            board_max_age_days=board_max_age_days,
            board_retention_days=board_retention_days,
            board_skip_prefix_map=board_skip_prefix_map,
            board_embed_batch_size=board_embed_batch_size,
            board_embed_retry_max=board_embed_retry_max,
            board_embed_backoff_base=board_embed_backoff_base,
            log_level=log_level,
            cloudflare_account_id=cloudflare_account_id,
            cloudflare_r2_access_key_id=cloudflare_r2_access_key_id,
            cloudflare_r2_secret_access_key=cloudflare_r2_secret_access_key,
            cloudflare_r2_bucket=cloudflare_r2_bucket,
            cloudflare_r2_endpoint=cloudflare_r2_endpoint,
            cloudflare_r2_key_prefix=cloudflare_r2_key_prefix,
            cloudflare_r2_soft_delete_enabled=cloudflare_r2_soft_delete_enabled,
            cloudflare_r2_soft_delete_prefix=cloudflare_r2_soft_delete_prefix,
        )

    def get_provider_config(self) -> ProviderConfig:
        """Get provider configuration for use with ProviderFactory"""
        return ProviderConfig(
            embedding_provider=self.embedding_provider,
            vector_provider=self.vector_provider,
        )

    def get_embedding_service_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for embedding service"""
        if self.embedding_provider == EmbeddingProvider.OPENAI:
            return {
                "api_key": self.openai_api_key,
                "model": self.openai_model,
                "base_url": self.openai_base_url,
                "max_tokens": self.max_tokens,
            }
        else:
            raise ValueError(
                f"Unsupported embedding provider: {self.embedding_provider}"
            )

    def get_vector_service_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for vector service"""
        if self.vector_provider == VectorProvider.QDRANT_CLOUD:
            return {"url": self.qdrant_cloud_url, "api_key": self.qdrant_api_key}
        else:
            raise ValueError(f"Unsupported vector provider: {self.vector_provider}")

    def validate(self) -> None:
        """Validate configuration based on selected providers"""
        # Validate embedding provider configuration
        if self.embedding_provider == EmbeddingProvider.OPENAI:
            if not self.openai_api_key or not self.openai_api_key.strip():
                raise ValueError(
                    "OpenAI API key is required when using OpenAI embedding provider"
                )

        # Validate vector provider configuration
        if self.vector_provider == VectorProvider.QDRANT_CLOUD:
            if not self.qdrant_api_key or not self.qdrant_api_key.strip():
                raise ValueError(
                    "Qdrant Cloud API key is required when using Qdrant Cloud provider"
                )
            if not self.qdrant_cloud_url or not self.qdrant_cloud_url.strip():
                raise ValueError(
                    "Qdrant Cloud URL is required when using Qdrant Cloud provider"
                )

    def validate_r2(self) -> None:
        """Validate Cloudflare R2 configuration before enabling sync."""
        required_fields = {
            "CLOUDFLARE_ACCOUNT_ID": self.cloudflare_account_id,
            "CLOUDFLARE_R2_ACCESS_KEY_ID": self.cloudflare_r2_access_key_id,
            "CLOUDFLARE_R2_SECRET_ACCESS_KEY": self.cloudflare_r2_secret_access_key,
            "CLOUDFLARE_R2_BUCKET": self.cloudflare_r2_bucket,
            "CLOUDFLARE_R2_ENDPOINT": self.cloudflare_r2_endpoint,
        }
        missing = [
            field for field, value in required_fields.items() if not value.strip()
        ]
        if missing:
            raise ValueError(
                "Missing Cloudflare R2 configuration values: " + ", ".join(missing)
            )

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            "repo_url": self.repo_url,
            "branch": self.branch,
            "repo_cache_dir": self.repo_cache_dir,
            "embedding_provider": str(self.embedding_provider),
            "vector_provider": str(self.vector_provider),
            "qdrant_collection": self.qdrant_collection,
            "vector_size": self.vector_size,
            "qdrant_cloud_url": self.qdrant_cloud_url,
            "qdrant_api_key": self.qdrant_api_key,
            "qdrant_cluster_region": self.qdrant_cluster_region,
            "openai_api_key": self.openai_api_key,
            "openai_model": self.openai_model,
            "openai_base_url": self.openai_base_url,
            "max_workers": self.max_workers,
            "max_document_chars": self.max_document_chars,
            "max_tokens": self.max_tokens,
            "chunk_threshold": self.chunk_threshold,
            "chunk_overlap": self.chunk_overlap,
            "board_retention_days": self.board_retention_days,
            "board_skip_prefix_map": {
                str(k): list(v) for k, v in self.board_skip_prefix_map.items()
            },
            "board_embed_batch_size": self.board_embed_batch_size,
            "board_embed_retry_max": self.board_embed_retry_max,
            "board_embed_backoff_base": self.board_embed_backoff_base,
            "log_level": self.log_level,
            "cloudflare_account_id": self.cloudflare_account_id,
            "cloudflare_r2_access_key_id": self.cloudflare_r2_access_key_id,
            "cloudflare_r2_bucket": self.cloudflare_r2_bucket,
            "cloudflare_r2_endpoint": self.cloudflare_r2_endpoint,
            "cloudflare_r2_key_prefix": self.cloudflare_r2_key_prefix,
            "cloudflare_r2_soft_delete_enabled": self.cloudflare_r2_soft_delete_enabled,
            "cloudflare_r2_soft_delete_prefix": self.cloudflare_r2_soft_delete_prefix,
        }

    def get_r2_service_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for Cloudflare R2 service"""
        return {
            "account_id": self.cloudflare_account_id,
            "access_key_id": self.cloudflare_r2_access_key_id,
            "secret_access_key": self.cloudflare_r2_secret_access_key,
            "bucket": self.cloudflare_r2_bucket,
            "endpoint": self.cloudflare_r2_endpoint,
            "key_prefix": self.cloudflare_r2_key_prefix,
            "soft_delete_enabled": self.cloudflare_r2_soft_delete_enabled,
            "soft_delete_prefix": self.cloudflare_r2_soft_delete_prefix,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary"""
        # Convert string providers back to enums
        if "embedding_provider" in config_dict:
            config_dict["embedding_provider"] = EmbeddingProvider(
                config_dict["embedding_provider"]
            )
        if "vector_provider" in config_dict:
            config_dict["vector_provider"] = VectorProvider(
                config_dict["vector_provider"]
            )

        return cls(**config_dict)
