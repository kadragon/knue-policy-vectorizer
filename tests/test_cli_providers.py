"""
Tests for CLI provider selection and configuration functionality
"""

import os
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from src.config import Config
from src.providers import EmbeddingProvider, VectorProvider


# We'll test the CLI commands after implementing them
class TestCLIProviders:
    """Test CLI provider selection functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.runner = CliRunner()

    def test_list_providers_command(self):
        """Test listing available providers"""
        # Import after implementing
        from src.sync_pipeline import list_providers

        result = self.runner.invoke(list_providers)
        assert result.exit_code == 0
        assert "Available Embedding Providers:" in result.output
        assert "Available Vector Providers:" in result.output
        assert "ollama" in result.output
        assert "openai" in result.output
        assert "qdrant_local" in result.output
        assert "qdrant_cloud" in result.output

    def test_configure_providers_interactive(self):
        """Test interactive provider configuration"""
        from src.sync_pipeline import configure_providers

        # Simulate user selecting OpenAI + Qdrant Cloud
        with patch("click.prompt") as mock_prompt:
            mock_prompt.side_effect = [
                "openai",  # Embedding provider
                "sk-test-key",  # OpenAI API key
                "text-embedding-3-large",  # OpenAI model
                "qdrant_cloud",  # Vector provider
                "https://test.qdrant.tech",  # Qdrant Cloud URL
                "test-api-key",  # Qdrant API key
            ]

            with patch("click.confirm", return_value=True):  # Confirm saving
                result = self.runner.invoke(configure_providers)

        assert result.exit_code == 0
        assert (
            "Configuration saved with" in result.output
        )  # Could be "masked credentials" or "secure permissions"

    def test_configure_providers_with_validation_errors(self):
        """Test configuration with validation errors"""
        from src.sync_pipeline import configure_providers

        # Test invalid provider selection
        with patch("click.prompt") as mock_prompt:
            mock_prompt.side_effect = [
                "invalid_provider",  # Invalid embedding provider
                "ollama",  # Valid fallback
                "bge-m3",  # Model
                "qdrant_local",  # Vector provider
                "http://localhost:6333",  # Qdrant URL
            ]

            with patch("click.confirm", return_value=True):
                result = self.runner.invoke(configure_providers)

        assert result.exit_code == 0
        assert (
            "Invalid provider" in result.output
            or "Configuration saved" in result.output
        )

    def test_show_config_command(self):
        """Test showing current configuration"""
        from src.sync_pipeline import show_config

        result = self.runner.invoke(show_config)
        assert result.exit_code == 0
        assert "Current Configuration" in result.output
        assert "Embedding Provider:" in result.output
        assert "Vector Provider:" in result.output

    def test_sync_with_provider_options(self):
        """Test sync command with provider options"""
        from src.sync_pipeline import main

        with patch("src.sync_pipeline.SyncPipeline") as mock_pipeline:
            mock_instance = Mock()
            mock_instance.health_check.return_value = True
            mock_instance.sync.return_value = {
                "status": "success",
                "changes_detected": False,
                "upserted": 0,
                "deleted": 0,
                "renamed": 0,
                "failed_files": [],
            }
            mock_pipeline.return_value = mock_instance

            result = self.runner.invoke(
                main,
                [
                    "sync",
                    "--embedding-provider",
                    "openai",
                    "--vector-provider",
                    "qdrant_cloud",
                    "--openai-api-key",
                    "sk-test",
                    "--qdrant-cloud-url",
                    "https://test.qdrant.tech",
                    "--qdrant-api-key",
                    "test-key",
                ],
            )

        assert result.exit_code == 0
        assert "No changes detected" in result.output

    def test_health_command_with_providers(self):
        """Test health command with different providers"""
        from src.sync_pipeline import main

        with patch("src.sync_pipeline.SyncPipeline") as mock_pipeline:
            mock_instance = Mock()
            mock_instance.health_check.return_value = True
            mock_pipeline.return_value = mock_instance

            result = self.runner.invoke(
                main,
                [
                    "health",
                    "--embedding-provider",
                    "openai",
                    "--vector-provider",
                    "qdrant_cloud",
                    "--openai-api-key",
                    "sk-test",
                    "--qdrant-cloud-url",
                    "https://test.qdrant.tech",
                    "--qdrant-api-key",
                    "test-key",
                ],
            )

        assert result.exit_code == 0
        assert "All services are healthy" in result.output

    def test_test_providers_command(self):
        """Test the test-providers command for connectivity validation"""
        from src.sync_pipeline import test_providers

        with patch("src.providers.ProviderFactory") as mock_factory:
            mock_factory_instance = Mock()
            mock_embedding_service = Mock()
            mock_vector_service = Mock()

            mock_embedding_service.health_check.return_value = True
            mock_vector_service.health_check.return_value = True

            mock_factory_instance.get_embedding_service.return_value = (
                mock_embedding_service
            )
            mock_factory_instance.get_vector_service.return_value = mock_vector_service
            mock_factory.return_value = mock_factory_instance

            result = self.runner.invoke(
                test_providers,
                [
                    "--embedding-provider",
                    "openai",
                    "--vector-provider",
                    "qdrant_cloud",
                    "--openai-api-key",
                    "sk-test",
                    "--qdrant-cloud-url",
                    "https://test.qdrant.tech",
                    "--qdrant-api-key",
                    "test-key",
                ],
            )

        assert result.exit_code == 0
        assert "Provider connectivity test completed" in result.output

    def test_migrate_providers_command(self):
        """Test the migrate command for switching providers"""
        from src.sync_pipeline import migrate_providers

        with patch("src.sync_pipeline.SyncPipeline") as mock_pipeline:
            mock_instance = Mock()
            mock_instance.health_check.return_value = True
            mock_pipeline.return_value = mock_instance

            with patch("click.confirm", return_value=True):  # Confirm migration
                result = self.runner.invoke(
                    migrate_providers,
                    [
                        "--from-embedding",
                        "ollama",
                        "--from-vector",
                        "qdrant_local",
                        "--to-embedding",
                        "openai",
                        "--to-vector",
                        "qdrant_cloud",
                    ],
                )

        assert result.exit_code == 0

    def test_config_file_operations(self):
        """Test configuration file save/load operations"""
        from src.sync_pipeline import load_config_file, save_config_file

        config = Config(
            embedding_provider=EmbeddingProvider.OPENAI,
            vector_provider=VectorProvider.QDRANT_CLOUD,
            openai_api_key="sk-test",
            qdrant_cloud_url="https://test.qdrant.tech",
        )

        with self.runner.isolated_filesystem():
            # Test saving config
            result = self.runner.invoke(
                save_config_file, ["--output", "test-config.env"], obj=config
            )

            assert result.exit_code == 0
            assert os.path.exists("test-config.env")

            # Test loading config
            result = self.runner.invoke(
                load_config_file, ["--config-file", "test-config.env"]
            )

            assert result.exit_code == 0

    def test_provider_validation_in_cli(self):
        """Test provider validation in CLI commands"""
        from src.sync_pipeline import main

        # Test with invalid provider
        result = self.runner.invoke(
            main, ["sync", "--embedding-provider", "invalid_provider"]
        )

        assert result.exit_code != 0
        assert "Invalid" in result.output or "Error" in result.output

    def test_environment_variable_override(self):
        """Test that CLI options override environment variables"""
        from src.sync_pipeline import main

        with patch.dict(
            os.environ,
            {"EMBEDDING_PROVIDER": "ollama", "VECTOR_PROVIDER": "qdrant_local"},
        ):
            with patch("src.sync_pipeline.SyncPipeline") as mock_pipeline:
                mock_instance = Mock()
                mock_instance.health_check.return_value = True
                mock_instance.sync.return_value = {
                    "status": "success",
                    "changes_detected": False,
                    "upserted": 0,
                    "deleted": 0,
                    "renamed": 0,
                    "failed_files": [],
                }
                mock_pipeline.return_value = mock_instance

                # CLI options should override env vars
                result = self.runner.invoke(
                    main,
                    [
                        "sync",
                        "--embedding-provider",
                        "openai",
                        "--openai-api-key",
                        "sk-test",
                    ],
                )

        assert result.exit_code == 0

    def test_config_export_import(self):
        """Test configuration export and import functionality"""
        from src.sync_pipeline import export_config, import_config

        with self.runner.isolated_filesystem():
            # Export current config
            result = self.runner.invoke(
                export_config, ["--format", "json", "--output", "config.json"]
            )

            assert result.exit_code == 0
            assert os.path.exists("config.json")

            # Import config
            result = self.runner.invoke(import_config, ["--config-file", "config.json"])

            assert result.exit_code == 0

    def test_provider_status_display(self):
        """Test provider status display in health command"""
        from src.sync_pipeline import main

        with patch("src.sync_pipeline.SyncPipeline") as mock_pipeline:
            mock_instance = Mock()
            mock_instance.health_check.return_value = True
            mock_instance.config = Config(
                embedding_provider=EmbeddingProvider.OPENAI,
                vector_provider=VectorProvider.QDRANT_CLOUD,
            )
            mock_pipeline.return_value = mock_instance

            result = self.runner.invoke(main, ["health", "--verbose"])

        assert result.exit_code == 0
        # Should show provider information in verbose mode
