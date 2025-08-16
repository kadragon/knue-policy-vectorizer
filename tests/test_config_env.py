"""Tests for Config.from_env() environment variable mapping."""

import os
import sys

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_config_from_env_supports_legacy_and_canonical_keys(monkeypatch):
    from config import Config

    # Clear potentially conflicting vars first
    keys_to_clear = [
        "GIT_REPO_URL",
        "REPO_URL",
        "GIT_BRANCH",
        "BRANCH",
        "QDRANT_URL",
        "COLLECTION_NAME",
        "QDRANT_COLLECTION",
        "VECTOR_SIZE",
        "OLLAMA_URL",
        "OLLAMA_MODEL",
        "EMBEDDING_MODEL",
        "MAX_WORKERS",
        "MAX_DOCUMENT_CHARS",
        "MAX_TOKEN_LENGTH",
        "MAX_TOKENS",
        "CHUNK_THRESHOLD",
        "CHUNK_OVERLAP",
        "LOG_LEVEL",
        "REPO_CACHE_DIR",
    ]
    for k in keys_to_clear:
        monkeypatch.delenv(k, raising=False)

    # Set canonical envs
    monkeypatch.setenv("GIT_REPO_URL", "https://example.com/repo.git")
    monkeypatch.setenv("GIT_BRANCH", "prod")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.local:6333")
    monkeypatch.setenv("COLLECTION_NAME", "policies")
    monkeypatch.setenv("VECTOR_SIZE", "1536")
    monkeypatch.setenv("OLLAMA_URL", "http://ollama.local:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "bge-test")
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
    assert cfg.qdrant_url == "http://qdrant.local:6333"
    assert cfg.qdrant_collection == "policies"
    assert cfg.vector_size == 1536
    assert cfg.ollama_url == "http://ollama.local:11434"
    assert cfg.embedding_model == "bge-test"
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
    monkeypatch.setenv("EMBEDDING_MODEL", "legacy-model")
    monkeypatch.setenv("MAX_TOKENS", "2048")

    cfg2 = Config.from_env()

    assert cfg2.repo_url == "https://legacy/repo.git"
    assert cfg2.branch == "legacy"
    assert cfg2.qdrant_collection == "legacy_collection"
    assert cfg2.embedding_model == "legacy-model"
    assert cfg2.max_tokens == 2048
