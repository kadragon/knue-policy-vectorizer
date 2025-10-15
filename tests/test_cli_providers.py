"""Tests for CLI provider selection and configuration functionality"""

import os
from unittest.mock import Mock, patch

import click
import pytest
from click.testing import CliRunner

from src.config.config import Config
from src.utils.providers import EmbeddingProvider, VectorProvider


# We'll test the CLI commands after implementing them
class TestCLIProviders:
    """Test CLI provider selection functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.runner = CliRunner()
        for key in [
            "EMBEDDING_PROVIDER",
            "VECTOR_PROVIDER",
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
            "OPENAI_BASE_URL",
            "QDRANT_CLOUD_URL",
            "QDRANT_API_KEY",
        ]:
            os.environ.pop(key, None)

    def test_list_providers_command(self):
        """Test listing available providers"""
        # Import after implementing
        from src.pipelines.sync_pipeline import list_providers

        result = self.runner.invoke(list_providers)
        assert result.exit_code == 0
        assert "Available Embedding Providers:" in result.output
        assert "Available Vector Providers:" in result.output
        assert "openai" in result.output
        assert "qdrant_cloud" in result.output

    def test_configure_providers_interactive(self):
        """Test interactive provider configuration"""
        from src.pipelines.sync_pipeline import configure_providers

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
            "Configuration saving to .env is disabled for security reasons"
            in result.output
        )

    def test_configure_providers_with_validation_errors(self):
        """Test configuration with validation errors"""
        from src.pipelines.sync_pipeline import configure_providers

        # Test invalid provider selection
        with patch("click.prompt") as mock_prompt:
            mock_prompt.side_effect = [
                "invalid_provider",  # Invalid embedding provider
                "openai",  # Valid fallback
                "sk-test-key",  # OpenAI API key
                "text-embedding-3-small",  # Model
                "qdrant_cloud",  # Vector provider
                "https://test.qdrant.tech",  # Qdrant URL
                "test-api-key",  # Qdrant API key
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
        from src.pipelines.sync_pipeline import show_config

        result = self.runner.invoke(show_config)
        assert result.exit_code == 0
        assert "Current Configuration" in result.output
        assert "Embedding Provider:" in result.output
        assert "Vector Provider:" in result.output

    def test_sync_with_provider_options(self):
        """Test sync command with provider options"""
        from src.pipelines.sync_pipeline import main

        env = {
            "EMBEDDING_PROVIDER": "openai",
            "VECTOR_PROVIDER": "qdrant_cloud",
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL": "text-embedding-3-small",
            "QDRANT_CLOUD_URL": "https://test.qdrant.tech",
            "QDRANT_API_KEY": "test-key",
        }

        with (
            patch.dict(os.environ, env, clear=False),
            patch("src.pipelines.sync_pipeline.SyncPipeline") as mock_pipeline,
        ):
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

    @patch("src.config.config.Config.from_env")
    @patch("src.services.cloudflare_r2_service.CloudflareR2Service")
    @patch("src.pipelines.sync_pipeline.CloudflareR2SyncPipeline")
    def test_sync_cloudflare_r2_command(
        self,
        mock_pipeline,
        mock_service,
        mock_config_from_env,
    ):
        """Test successful Cloudflare R2 sync command."""
        from src.pipelines.sync_pipeline import sync_cloudflare_r2

        config = Config()
        config.cloudflare_r2_account_id = "account123"
        config.cloudflare_r2_bucket = "knue-vectorstore"
        config.cloudflare_r2_endpoint = "https://account123.r2.cloudflarestorage.com"
        config.validate_r2 = Mock()

        mock_config_from_env.return_value = config
        mock_instance = Mock()
        mock_instance.sync.return_value = {
            "status": "success",
            "changes_detected": True,
            "uploaded": 2,
            "deleted": 1,
            "renamed": 0,
            "failed_files": [],
        }
        mock_pipeline.return_value = mock_instance

        sync_cloudflare_r2.callback()
        config.validate_r2.assert_called_once()
        mock_pipeline.assert_called_once()

    @patch("src.config.config.Config.from_env")
    @patch("src.services.cloudflare_r2_service.CloudflareR2Service")
    @patch("src.pipelines.sync_pipeline.CloudflareR2SyncPipeline")
    def test_sync_cloudflare_r2_partial_failure_returns_nonzero_exit(
        self,
        mock_pipeline,
        mock_service,
        mock_config_from_env,
    ):
        """Partial sync failures should surface as errors."""
        import click

        from src.pipelines.sync_pipeline import sync_cloudflare_r2

        config = Config()
        config.cloudflare_r2_account_id = "account123"
        config.cloudflare_r2_bucket = "knue-vectorstore"
        config.cloudflare_r2_endpoint = "https://account123.r2.cloudflarestorage.com"
        config.validate_r2 = Mock()

        mock_config_from_env.return_value = config

        mock_instance = Mock()
        mock_instance.sync.return_value = {
            "status": "partial_success",
            "changes_detected": True,
            "uploaded": 1,
            "deleted": 0,
            "renamed": 0,
            "failed_files": ["file1.md"],
        }
        mock_pipeline.return_value = mock_instance

        with pytest.raises(click.ClickException):
            sync_cloudflare_r2.callback()

        config.validate_r2.assert_called_once()
        mock_pipeline.assert_called_once()

    @patch("src.pipelines.sync_pipeline.SyncPipeline")
    @patch("src.config.config.Config.from_env")
    def test_sync_partial_failure_returns_nonzero_exit(
        self,
        mock_config_from_env,
        mock_pipeline,
    ):
        """Partial failures in main sync command should exit non-zero."""
        from src.pipelines.sync_pipeline import sync

        config = Config()
        config.openai_api_key = "sk-test"
        config.qdrant_api_key = "qdrant-key"
        config.qdrant_cloud_url = "https://test.qdrant.tech"

        mock_config_from_env.return_value = config

        mock_instance = Mock()
        mock_instance.health_check.return_value = True
        mock_instance.sync.return_value = {
            "status": "partial_success",
            "changes_detected": True,
            "upserted": 1,
            "deleted": 0,
            "renamed": 0,
            "failed_files": ["file1.md"],
        }
        mock_pipeline.return_value = mock_instance

        with pytest.raises(click.ClickException):
            sync.callback()

        mock_instance.sync.assert_called_once()

    def test_health_command_with_providers(self):
        """Test health command with different providers"""
        from src.pipelines.sync_pipeline import main

        with patch("src.pipelines.sync_pipeline.SyncPipeline") as mock_pipeline:
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
        from src.pipelines.sync_pipeline import test_providers

        with patch("src.utils.providers.ProviderFactory") as mock_factory:
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
        from src.pipelines.sync_pipeline import migrate_providers

        with patch("src.pipelines.sync_pipeline.SyncPipeline") as mock_pipeline:
            mock_instance = Mock()
            mock_instance.health_check.return_value = True
            mock_pipeline.return_value = mock_instance

            with patch("click.confirm", return_value=True):  # Confirm migration
                result = self.runner.invoke(
                    migrate_providers,
                    [
                        "--from-embedding",
                        "openai",
                        "--from-vector",
                        "qdrant_cloud",
                        "--to-embedding",
                        "openai",
                        "--to-vector",
                        "qdrant_cloud",
                    ],
                )

        assert result.exit_code == 0

    def test_config_file_operations(self):
        """Test configuration file save/load operations"""
        from src.pipelines.sync_pipeline import load_config_file

        config = Config(
            embedding_provider=EmbeddingProvider.OPENAI,
            vector_provider=VectorProvider.QDRANT_CLOUD,
            openai_api_key="sk-test",
            qdrant_cloud_url="https://test.qdrant.tech",
        )

        # .env saving disabled, test loading only
        with self.runner.isolated_filesystem():
            # Create a test .env file manually
            with open("test-config.env", "w") as f:
                f.write("EMBEDDING_PROVIDER=openai\nVECTOR_PROVIDER=qdrant_cloud\n")

            # Test loading config
            result = self.runner.invoke(
                load_config_file, ["--config-file", "test-config.env"]
            )

            assert result.exit_code == 0

    def test_provider_validation_in_cli(self):
        """Test provider validation in CLI commands"""
        from src.pipelines.sync_pipeline import main

        # Test with invalid provider
        result = self.runner.invoke(
            main, ["sync", "--embedding-provider", "invalid_provider"]
        )

        assert result.exit_code != 0
        assert "Invalid" in result.output or "Error" in result.output

    def test_environment_variable_override(self):
        """Test that CLI options override environment variables"""
        from src.pipelines.sync_pipeline import main

        env = {
            "EMBEDDING_PROVIDER": "openai",
            "VECTOR_PROVIDER": "qdrant_cloud",
            "OPENAI_API_KEY": "sk-env",
            "OPENAI_MODEL": "text-embedding-3-small",
            "QDRANT_CLOUD_URL": "https://env.qdrant.tech",
            "QDRANT_API_KEY": "env-key",
        }

        with patch.dict(os.environ, env, clear=False):
            with patch("src.pipelines.sync_pipeline.SyncPipeline") as mock_pipeline:
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

    def test_config_import(self):
        """Test configuration import functionality"""
        from src.pipelines.sync_pipeline import import_config

        with self.runner.isolated_filesystem():
            # Create a test JSON config
            import json

            config_data = {
                "embedding_provider": "openai",
                "vector_provider": "qdrant_cloud",
                "qdrant_collection": "test_collection",
                "vector_size": 1536,
                "openai_api_key": "sk-test",
                "openai_model": "text-embedding-3-small",
                "qdrant_cloud_url": "https://test.qdrant.tech",
                "qdrant_api_key": "test-key",
            }
            with open("config.json", "w") as f:
                json.dump(config_data, f)

            # Import config
            result = self.runner.invoke(import_config, ["--config-file", "config.json"])

            assert result.exit_code == 0

    def test_provider_status_display(self):
        """Test provider status display in health command"""
        from src.pipelines.sync_pipeline import main

        with patch("src.pipelines.sync_pipeline.SyncPipeline") as mock_pipeline:
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
