"""Tests for Config.from_env() environment variable mapping."""

import os
import sys

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def test_config_from_env_supports_legacy_and_canonical_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config import Config

    # Clear potentially conflicting vars first
    keys_to_clear = [
        "GIT_REPO_URL",
        "REPO_URL",
        "GIT_BRANCH",
        "BRANCH",
        "COLLECTION_NAME",
        "QDRANT_COLLECTION",
        "VECTOR_SIZE",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
        "QDRANT_CLOUD_URL",
        "QDRANT_API_KEY",
        "MAX_WORKERS",
        "MAX_DOCUMENT_CHARS",
        "MAX_TOKEN_LENGTH",
        "MAX_TOKENS",
        "CHUNK_THRESHOLD",
        "CHUNK_OVERLAP",
        "LOG_LEVEL",
        "REPO_CACHE_DIR",
        "EMBEDDING_PROVIDER",
        "VECTOR_PROVIDER",
    ]
    for k in keys_to_clear:
        monkeypatch.delenv(k, raising=False)

    # Set canonical envs
    monkeypatch.setenv("GIT_REPO_URL", "https://example.com/repo.git")
    monkeypatch.setenv("GIT_BRANCH", "prod")
    monkeypatch.setenv("QDRANT_CLOUD_URL", "https://cloud.qdrant.tech")
    monkeypatch.setenv("QDRANT_API_KEY", "test-api-key")
    monkeypatch.setenv("COLLECTION_NAME", "policies")
    monkeypatch.setenv("VECTOR_SIZE", "1536")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("MAX_WORKERS", "8")
    monkeypatch.setenv("MAX_DOCUMENT_CHARS", "12345")
    monkeypatch.setenv("MAX_TOKEN_LENGTH", "4096")
    monkeypatch.setenv("CHUNK_THRESHOLD", "900")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("REPO_CACHE_DIR", "./cache")

    cfg = Config.from_env()

    assert cfg.repo_url == "https://example.com/repo.git"
    assert cfg.branch == "prod"
    assert cfg.qdrant_collection == "policies"
    assert cfg.vector_size == 1536
    assert cfg.qdrant_cloud_url == "https://cloud.qdrant.tech"
    assert cfg.qdrant_api_key == "test-api-key"
    assert cfg.openai_model == "text-embedding-3-small"
    assert cfg.openai_base_url == "https://api.example.com/v1"
    assert cfg.max_workers == 8
    assert cfg.max_document_chars == 12345
    assert cfg.max_tokens == 4096
    assert cfg.chunk_threshold == 900
    assert cfg.chunk_overlap == 100
    assert cfg.log_level == "DEBUG"
    assert cfg.repo_cache_dir == "./cache"

    # Now test legacy fallbacks
    for k in keys_to_clear:
        monkeypatch.delenv(k, raising=False)

    monkeypatch.setenv("REPO_URL", "https://legacy/repo.git")
    monkeypatch.setenv("BRANCH", "legacy")
    monkeypatch.setenv("QDRANT_COLLECTION", "legacy_collection")
    monkeypatch.setenv("MAX_TOKENS", "2048")

    cfg2 = Config.from_env()

    assert cfg2.repo_url == "https://legacy/repo.git"
    assert cfg2.branch == "legacy"
    assert cfg2.qdrant_collection == "legacy_collection"
    assert cfg2.max_tokens == 2048


def test_canonical_overrides_legacy_when_both_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config import Config

    # Clear
    for k in [
        "GIT_REPO_URL",
        "REPO_URL",
        "GIT_BRANCH",
        "BRANCH",
        "COLLECTION_NAME",
        "QDRANT_COLLECTION",
        "OPENAI_MODEL",
        "MAX_TOKEN_LENGTH",
        "MAX_TOKENS",
        "EMBEDDING_PROVIDER",
        "VECTOR_PROVIDER",
        "OPENAI_API_KEY",
        "QDRANT_CLOUD_URL",
        "QDRANT_API_KEY",
    ]:
        monkeypatch.delenv(k, raising=False)

    # Set both canonical and legacy
    monkeypatch.setenv("GIT_REPO_URL", "https://canonical/repo.git")
    monkeypatch.setenv("REPO_URL", "https://legacy/repo.git")

    monkeypatch.setenv("GIT_BRANCH", "canonical-branch")
    monkeypatch.setenv("BRANCH", "legacy-branch")

    monkeypatch.setenv("COLLECTION_NAME", "canonical_collection")
    monkeypatch.setenv("QDRANT_COLLECTION", "legacy_collection")

    monkeypatch.setenv("OPENAI_MODEL", "canonical-model")

    monkeypatch.setenv("MAX_TOKEN_LENGTH", "9999")
    monkeypatch.setenv("MAX_TOKENS", "1111")

    cfg = Config.from_env()

    assert cfg.repo_url == "https://canonical/repo.git"
    assert cfg.branch == "canonical-branch"
    assert cfg.qdrant_collection == "canonical_collection"
    assert cfg.openai_model == "canonical-model"
    assert cfg.max_tokens == 9999


def test_invalid_integer_env_values_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config import Config

    # Ensure a clean env
    monkeypatch.delenv("VECTOR_SIZE", raising=False)
    # Set invalid integer for a numeric field
    monkeypatch.setenv("VECTOR_SIZE", "not_an_int")

    with pytest.raises(ValueError):
        _ = Config.from_env()


def test_r2_configuration_loaded_and_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config import Config

    # Clear any existing R2 envs and provider envs
    for key in [
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_R2_ACCESS_KEY_ID",
        "CLOUDFLARE_R2_SECRET_ACCESS_KEY",
        "CLOUDFLARE_R2_BUCKET",
        "CLOUDFLARE_R2_ENDPOINT",
        "CLOUDFLARE_R2_KEY_PREFIX",
        "EMBEDDING_PROVIDER",
        "VECTOR_PROVIDER",
        "OPENAI_API_KEY",
        "QDRANT_CLOUD_URL",
        "QDRANT_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account123")
    monkeypatch.setenv("CLOUDFLARE_R2_ACCESS_KEY_ID", "access123")
    monkeypatch.setenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY", "secret456")
    monkeypatch.setenv("CLOUDFLARE_R2_BUCKET", "knue-vectorstore")
    monkeypatch.setenv(
        "CLOUDFLARE_R2_ENDPOINT",
        "https://account123.r2.cloudflarestorage.com/knue-vectorstore",
    )
    monkeypatch.setenv("CLOUDFLARE_R2_KEY_PREFIX", "policies")

    cfg = Config.from_env()
    assert cfg.cloudflare_account_id == "account123"
    assert cfg.cloudflare_r2_access_key_id == "access123"
    assert cfg.cloudflare_r2_bucket == "knue-vectorstore"
    assert cfg.cloudflare_r2_key_prefix == "policies"

    # Validation should pass with all values present
    cfg.validate_r2()

    # Remove one value and expect validation error
    monkeypatch.delenv("CLOUDFLARE_R2_BUCKET", raising=False)
    cfg_missing = Config.from_env()
    with pytest.raises(ValueError):
        cfg_missing.validate_r2()
