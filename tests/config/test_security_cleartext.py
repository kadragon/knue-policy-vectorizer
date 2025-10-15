"""
Tests for clear-text storage security fixes.

This module tests that sensitive data like API keys are properly masked
by default and only exposed when explicitly requested with warnings.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from src.config.config import Config
from src.config.config_manager import ConfigurationManager
from src.utils.providers import EmbeddingProvider, VectorProvider


class TestSecurityClearText:
    """Test security measures for sensitive data storage."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = Config(
            openai_api_key="sk-test123456789abcdef",
            qdrant_api_key="qd-secret987654321xyz",
            embedding_provider=EmbeddingProvider.OPENAI,
            vector_provider=VectorProvider.QDRANT_CLOUD,
        )
        self.config_manager = ConfigurationManager()

    def test_config_manager_export_masks_secrets_by_default(self):
        """Test that ConfigurationManager.export_config masks secrets by default."""
        # Test JSON format
        json_content = self.config_manager.export_config(
            self.config, format="json", include_secrets=False
        )
        assert "***MASKED***" in json_content
        assert "sk-test123456789abcdef" not in json_content
        assert "qd-secret987654321xyz" not in json_content

        # Test ENV format
        env_content = self.config_manager.export_config(
            self.config, format="env", include_secrets=False
        )
        assert (
            "***MASKED***" in env_content
            or "WARNING: Sensitive values are masked" in env_content
        )
        assert "sk-test123456789abcdef" not in env_content
        assert "qd-secret987654321xyz" not in env_content

    def test_config_manager_export_includes_secrets_when_requested(self):
        """Test that ConfigurationManager.export_config includes secrets when requested."""
        # Test JSON format
        json_content = self.config_manager.export_config(
            self.config, format="json", include_secrets=True
        )
        assert "sk-test123456789abcdef" in json_content
        assert "qd-secret987654321xyz" in json_content
        assert "***MASKED***" not in json_content

        # Test ENV format
        env_content = self.config_manager.export_config(
            self.config, format="env", include_secrets=True
        )
        assert "sk-test123456789abcdef" in env_content
        assert "qd-secret987654321xyz" in env_content
        assert "WARNING: This file contains sensitive credentials" in env_content


if __name__ == "__main__":
    pytest.main([__file__])
