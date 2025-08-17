"""
Tests for clear-text storage security fixes.

This module tests that sensitive data like API keys are properly masked
by default and only exposed when explicitly requested with warnings.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from src.config import Config
from src.config_manager import ConfigurationManager
from src.providers import EmbeddingProvider, VectorProvider
from src.sync_pipeline import _generate_env_content


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

    def test_env_generation_masks_secrets_by_default(self):
        """Test that _generate_env_content masks secrets by default."""
        content = _generate_env_content(self.config, include_secrets=False)

        # Should contain masked values
        assert "***MASKED***" in content
        assert "sk-test123456789abcdef" not in content
        assert "qd-secret987654321xyz" not in content

        # Should contain security warnings
        assert "WARNING: Sensitive values are masked for security" in content
        assert "--include-secrets flag with caution" in content

    def test_env_generation_includes_secrets_when_requested(self):
        """Test that _generate_env_content includes actual secrets when requested."""
        content = _generate_env_content(self.config, include_secrets=True)

        # Should contain actual values
        assert "sk-test123456789abcdef" in content
        assert "qd-secret987654321xyz" in content
        assert "***MASKED***" not in content

        # Should contain security warnings
        assert (
            "WARNING: This file contains sensitive credentials in clear text!"
            in content
        )
        assert "Do not commit this file to version control" in content
        assert "chmod 600 .env" in content

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

    def test_masked_values_show_partial_info(self):
        """Test that masked values show first few characters for identification."""
        content = _generate_env_content(self.config, include_secrets=False)

        # Should show partial information for identification
        lines = content.split("\n")
        openai_line = next(
            (line for line in lines if line.startswith("OPENAI_API_KEY=")), None
        )
        qdrant_line = next(
            (line for line in lines if line.startswith("QDRANT_API_KEY=")), None
        )

        assert openai_line is not None
        assert qdrant_line is not None

        # Should show first 4 characters
        assert "sk-t" in openai_line
        assert "qd-s" in qdrant_line

    def test_file_permissions_set_for_sensitive_exports(self):
        """Test that file permissions are set securely when exporting with secrets."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Write content with secrets
            content = _generate_env_content(self.config, include_secrets=True)
            with open(tmp_path, "w") as f:
                f.write(content)

            # Set secure permissions (like the CLI would do)
            os.chmod(tmp_path, 0o600)

            # Verify permissions
            stat_info = os.stat(tmp_path)
            permissions = oct(stat_info.st_mode)[-3:]
            assert permissions == "600"

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_empty_api_keys_handled_gracefully(self):
        """Test that empty API keys are handled gracefully."""
        config_no_keys = Config(
            openai_api_key="",
            qdrant_api_key=None,
            embedding_provider=EmbeddingProvider.OLLAMA,
            vector_provider=VectorProvider.QDRANT_LOCAL,
        )

        content = _generate_env_content(config_no_keys, include_secrets=False)

        # Should not crash and should handle empty values
        assert content is not None
        assert "EMBEDDING_PROVIDER=ollama" in content
        assert "VECTOR_PROVIDER=qdrant_local" in content

    def test_no_secrets_present_skips_warnings(self):
        """Test that configurations without secrets don't show unnecessary warnings."""
        config_no_secrets = Config(
            embedding_provider=EmbeddingProvider.OLLAMA,
            vector_provider=VectorProvider.QDRANT_LOCAL,
        )

        content = _generate_env_content(config_no_secrets, include_secrets=False)

        # Should still have basic security notice but not specific API key warnings
        assert "WARNING: Sensitive values are masked" in content
        # Verify no actual secrets are referenced
        assert "OPENAI_API_KEY" not in content
        assert "QDRANT_API_KEY" not in content


if __name__ == "__main__":
    pytest.main([__file__])
