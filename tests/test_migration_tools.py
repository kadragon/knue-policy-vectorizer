"""
Tests for migration tools functionality
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.config import Config
from src.migration_tools import (
    CompatibilityCheck,
    MigrationManager,
    MigrationReport,
    create_migration_config,
)
from src.providers import EmbeddingProvider, VectorProvider


class TestMigrationReport:
    """Test MigrationReport functionality"""

    def test_migration_report_creation(self):
        """Test creating migration report"""
        start_time = datetime.now()
        report = MigrationReport(
            source_provider="openai/qdrant_cloud",
            target_provider="openai/qdrant_cloud",
            start_time=start_time,
            total_documents=100,
            migrated_documents=95,
            failed_documents=5,
        )

        assert report.source_provider == "openai/qdrant_cloud"
        assert report.target_provider == "openai/qdrant_cloud"
        assert report.total_documents == 100
        assert report.migrated_documents == 95
        assert report.failed_documents == 5
        assert report.success_rate == 95.0

    def test_migration_report_to_dict(self):
        """Test exporting migration report as dictionary"""
        start_time = datetime.now()
        end_time = datetime.now()

        report = MigrationReport(
            source_provider="openai/qdrant_cloud",
            target_provider="openai/qdrant_cloud",
            start_time=start_time,
            end_time=end_time,
            total_documents=100,
            migrated_documents=95,
            failed_documents=5,
            errors=["Test error"],
            performance_metrics={"avg_time": 0.1},
        )

        report_dict = report.to_dict()

        assert report_dict["source_provider"] == "openai/qdrant_cloud"
        assert report_dict["target_provider"] == "openai/qdrant_cloud"
        assert report_dict["total_documents"] == 100
        assert report_dict["migrated_documents"] == 95
        assert report_dict["failed_documents"] == 5
        assert report_dict["success_rate"] == 95.0
        assert report_dict["errors"] == ["Test error"]
        assert report_dict["performance_metrics"] == {"avg_time": 0.1}


class TestCompatibilityCheck:
    """Test CompatibilityCheck functionality"""

    def test_compatibility_check_creation(self):
        """Test creating compatibility check"""
        check = CompatibilityCheck(
            embedding_compatible=True,
            vector_compatible=True,
            dimension_match=True,
            source_dimensions=1536,
            target_dimensions=1536,
            warnings=["Test warning"],
        )

        assert check.embedding_compatible is True
        assert check.vector_compatible is True
        assert check.dimension_match is True
        assert check.source_dimensions == 1536
        assert check.target_dimensions == 1536
        assert check.fully_compatible is True
        assert check.warnings == ["Test warning"]

    def test_compatibility_check_not_fully_compatible(self):
        """Test compatibility check when not fully compatible"""
        check = CompatibilityCheck(
            embedding_compatible=True,
            vector_compatible=True,
            dimension_match=False,
            source_dimensions=1536,
            target_dimensions=2048,
        )

        assert check.fully_compatible is False


class TestMigrationManager:
    """Test MigrationManager functionality"""

    def setup_method(self):
        """Setup test environment"""
        # Create test configurations
        self.source_config = Config(
            embedding_provider=EmbeddingProvider.OPENAI,
            vector_provider=VectorProvider.QDRANT_CLOUD,
            openai_api_key="sk-test-source",
            openai_model="text-embedding-3-small",
            qdrant_cloud_url="https://source.qdrant.tech",
            qdrant_api_key="source-key",
            qdrant_collection="test_collection",
            vector_size=1536,
        )

        self.target_config = Config(
            embedding_provider=EmbeddingProvider.OPENAI,
            vector_provider=VectorProvider.QDRANT_CLOUD,
            openai_api_key="sk-test",
            openai_model="text-embedding-3-small",
            qdrant_cloud_url="https://test.qdrant.tech",
            qdrant_api_key="test-key",
            qdrant_collection="test_collection",
            vector_size=1536,
        )

    def test_migration_manager_initialization(self):
        """Test migration manager initialization"""
        with patch("src.migration_tools.ProviderFactory"):
            manager = MigrationManager(self.source_config, self.target_config)

            assert manager.source_config == self.source_config
            assert manager.target_config == self.target_config

    @patch("src.migration_tools.ProviderFactory")
    def test_check_compatibility_success(self, mock_factory):
        """Test compatibility check with matching dimensions"""
        # Setup mocks
        mock_factory_instance = Mock()
        mock_factory.return_value = mock_factory_instance

        mock_source_embedding = Mock()
        mock_target_embedding = Mock()
        mock_source_vector = Mock()
        mock_target_vector = Mock()

        # Mock embedding services
        mock_source_embedding.generate_embedding.return_value = [0.1] * 1536
        mock_target_embedding.generate_embedding.return_value = [0.2] * 1536

        # Mock vector services
        mock_source_vector.health_check.return_value = True
        mock_target_vector.health_check.return_value = True

        mock_factory_instance.get_embedding_service.side_effect = [
            mock_source_embedding,
            mock_target_embedding,
        ]
        mock_factory_instance.get_vector_service.side_effect = [
            mock_source_vector,
            mock_target_vector,
        ]

        manager = MigrationManager(self.source_config, self.target_config)
        compatibility = manager.check_compatibility()

        assert compatibility.embedding_compatible is True
        assert compatibility.vector_compatible is True
        assert compatibility.dimension_match is True
        assert compatibility.source_dimensions == 1536
        assert compatibility.target_dimensions == 1536

    @patch("src.migration_tools.ProviderFactory")
    def test_check_compatibility_dimension_mismatch(self, mock_factory):
        """Test compatibility check with dimension mismatch"""
        # Setup mocks
        mock_factory_instance = Mock()
        mock_factory.return_value = mock_factory_instance

        mock_source_embedding = Mock()
        mock_target_embedding = Mock()
        mock_source_vector = Mock()
        mock_target_vector = Mock()

        # Mock embedding services with different dimensions
        mock_source_embedding.generate_embedding.return_value = [0.1] * 1536
        mock_target_embedding.generate_embedding.return_value = [0.2] * 2048

        # Mock vector services
        mock_source_vector.health_check.return_value = True
        mock_target_vector.health_check.return_value = True

        mock_factory_instance.get_embedding_service.side_effect = [
            mock_source_embedding,
            mock_target_embedding,
        ]
        mock_factory_instance.get_vector_service.side_effect = [
            mock_source_vector,
            mock_target_vector,
        ]

        # Force vector size mismatch for target
        original_target_size = self.target_config.vector_size
        self.target_config.vector_size = 2048
        manager = MigrationManager(self.source_config, self.target_config)
        compatibility = manager.check_compatibility()

        assert compatibility.embedding_compatible is True
        assert compatibility.vector_compatible is True
        assert compatibility.dimension_match is False
        assert compatibility.source_dimensions == 1536
        assert compatibility.target_dimensions == 2048
        assert len(compatibility.warnings) > 0

        # Restore target vector size for other tests
        self.target_config.vector_size = original_target_size
        assert "Dimension mismatch" in compatibility.warnings[0]

    @patch("src.migration_tools.ProviderFactory")
    def test_create_backup_success(self, mock_factory):
        """Test successful backup creation"""
        # Setup mocks
        mock_factory_instance = Mock()
        mock_factory.return_value = mock_factory_instance

        mock_vector_service = Mock()
        mock_vector_service.search_points.return_value = [
            {
                "id": "doc1",
                "vector": [0.1] * 1536,
                "payload": {"title": "Test Document 1", "content": "Test content 1"},
            },
            {
                "id": "doc2",
                "vector": [0.2] * 1536,
                "payload": {"title": "Test Document 2", "content": "Test content 2"},
            },
        ]

        mock_factory_instance.get_vector_service.return_value = mock_vector_service

        manager = MigrationManager(self.source_config, self.target_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            backup_path = f"{temp_dir}/test_backup.json"
            result = manager.create_backup(backup_path)

            assert result["success"] is True
            assert result["points_backed_up"] == 2
            assert "backup_path" in result
            assert "file_size" in result

            # Verify backup file was created and contains correct data
            assert Path(backup_path).exists()
            with open(backup_path, "r") as f:
                backup_data = json.load(f)

            assert len(backup_data["points"]) == 2
            assert backup_data["total_points"] == 2
            assert backup_data["collection_name"] == "test_collection"

    @patch("src.migration_tools.ProviderFactory")
    def test_restore_from_backup_success(self, mock_factory):
        """Test successful restore from backup"""
        # Setup mocks
        mock_factory_instance = Mock()
        mock_factory.return_value = mock_factory_instance

        mock_vector_service = Mock()
        mock_vector_service.collection_exists.return_value = False
        mock_vector_service.create_collection.return_value = True
        mock_vector_service.upsert_points.return_value = True

        mock_factory_instance.get_vector_service.return_value = mock_vector_service

        manager = MigrationManager(self.source_config, self.target_config)

        # Create test backup file
        backup_data = {
            "created_at": datetime.now().isoformat(),
            "source_config": self.source_config.to_dict(),
            "collection_name": "test_collection",
            "total_points": 2,
            "points": [
                {
                    "id": "doc1",
                    "vector": [0.1] * 1536,
                    "payload": {
                        "title": "Test Document 1",
                        "content": "Test content 1",
                    },
                },
                {
                    "id": "doc2",
                    "vector": [0.2] * 1536,
                    "payload": {
                        "title": "Test Document 2",
                        "content": "Test content 2",
                    },
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(backup_data, f)
            backup_path = f.name

        try:
            result = manager.restore_from_backup(backup_path)

            assert result["success"] is True
            assert result["points_restored"] == 2
            assert result["collection"] == "test_collection"

            # Verify collection was created and points were upserted
            mock_vector_service.create_collection.assert_called_once()
            mock_vector_service.upsert_points.assert_called()

        finally:
            Path(backup_path).unlink()

    @patch("src.migration_tools.ProviderFactory")
    def test_compare_performance(self, mock_factory):
        """Test performance comparison between providers"""
        # Setup mocks
        mock_factory_instance = Mock()
        mock_factory.return_value = mock_factory_instance

        mock_source_embedding = Mock()
        mock_target_embedding = Mock()
        mock_source_vector = Mock()
        mock_target_vector = Mock()

        # Mock embedding performance (source slower than target)
        mock_source_embedding.generate_embeddings_batch.return_value = [
            [0.1] * 1536,
            [0.2] * 1536,
        ]
        mock_target_embedding.generate_embeddings_batch.return_value = [
            [0.3] * 1536,
            [0.4] * 1536,
        ]

        # Mock vector performance
        mock_source_vector.collection_exists.return_value = True
        mock_target_vector.collection_exists.return_value = True
        mock_source_vector.upsert_points.return_value = True
        mock_target_vector.upsert_points.return_value = True
        mock_source_vector.delete_points.return_value = True
        mock_target_vector.delete_points.return_value = True

        mock_factory_instance.get_embedding_service.side_effect = [
            mock_source_embedding,
            mock_target_embedding,
        ]
        mock_factory_instance.get_vector_service.side_effect = [
            mock_source_vector,
            mock_target_vector,
        ]

        manager = MigrationManager(self.source_config, self.target_config)

        test_texts = ["Test text 1", "Test text 2"]
        comparison = manager.compare_performance(test_texts)

        assert comparison["test_size"] == 2
        assert "embedding_performance" in comparison
        assert "vector_performance" in comparison
        assert "source_provider" in comparison
        assert "target_provider" in comparison

        # Verify embedding performance was measured
        emb_perf = comparison["embedding_performance"]
        assert "source" in emb_perf
        assert "target" in emb_perf
        assert "speedup" in emb_perf
        assert emb_perf["source"]["dimensions"] == 1536
        assert emb_perf["target"]["dimensions"] == 1536


class TestCreateMigrationConfig:
    """Test migration configuration creation"""

    def test_create_migration_config_basic(self):
        """Test creating basic migration configuration"""
        with patch("src.migration_tools.Config") as mock_config_class:
            mock_base_config = Mock()
            mock_base_config.to_dict.return_value = {
                "repo_url": "test",
                "branch": "main",
                "qdrant_collection": "test",
            }
            mock_config_class.from_env.return_value = mock_base_config
            mock_config_class.from_dict.side_effect = lambda d: d

            source_config, target_config = create_migration_config(
                "openai", "qdrant_cloud", "openai", "qdrant_cloud"
            )

            # Verify configurations were created with correct providers
            assert mock_config_class.from_dict.call_count == 2

            # Check the calls made to from_dict
            calls = mock_config_class.from_dict.call_args_list
            source_dict = calls[0][0][0]
            target_dict = calls[1][0][0]

            assert source_dict["embedding_provider"] == EmbeddingProvider.OPENAI
            assert source_dict["vector_provider"] == VectorProvider.QDRANT_CLOUD
            assert target_dict["embedding_provider"] == EmbeddingProvider.OPENAI
            assert target_dict["vector_provider"] == VectorProvider.QDRANT_CLOUD

    def test_create_migration_config_with_overrides(self):
        """Test creating migration configuration with overrides"""
        with patch("src.migration_tools.Config") as mock_config_class:
            mock_base_config = Mock()
            mock_base_config.to_dict.return_value = {
                "repo_url": "test",
                "branch": "main",
                "qdrant_collection": "test",
            }
            mock_config_class.from_env.return_value = mock_base_config
            mock_config_class.from_dict.side_effect = lambda d: d

            source_overrides = {"openai_api_key": "source-key"}
            target_overrides = {"openai_api_key": "target-key"}

            source_config, target_config = create_migration_config(
                "openai",
                "qdrant_cloud",
                "openai",
                "qdrant_cloud",
                source_overrides=source_overrides,
                target_overrides=target_overrides,
            )

            # Check that overrides were applied
            calls = mock_config_class.from_dict.call_args_list
            source_dict = calls[0][0][0]
            target_dict = calls[1][0][0]

            assert source_dict["openai_api_key"] == "source-key"
            assert target_dict["openai_api_key"] == "target-key"


class TestMigrationIntegration:
    """Test migration tools integration scenarios"""

    @patch("src.migration_tools.ProviderFactory")
    def test_full_migration_workflow(self, mock_factory):
        """Test complete migration workflow"""
        # Setup comprehensive mocks
        mock_factory_instance = Mock()
        mock_factory.return_value = mock_factory_instance

        # Source services
        mock_source_embedding = Mock()
        mock_source_vector = Mock()

        # Target services
        mock_target_embedding = Mock()
        mock_target_vector = Mock()

        # Configure embedding services
        mock_source_embedding.generate_embedding.return_value = [0.1] * 1536
        mock_target_embedding.generate_embedding.return_value = [0.2] * 1536
        mock_target_embedding.generate_embeddings_batch.return_value = [
            [0.3] * 1536,
            [0.4] * 1536,
        ]

        # Configure vector services
        mock_source_vector.health_check.return_value = True
        mock_target_vector.health_check.return_value = True
        mock_source_vector.search_points.return_value = [
            {
                "id": "doc1",
                "vector": [0.1] * 1536,
                "payload": {"content": "Test document 1", "title": "Doc 1"},
            },
            {
                "id": "doc2",
                "vector": [0.2] * 1536,
                "payload": {"content": "Test document 2", "title": "Doc 2"},
            },
        ]
        mock_target_vector.collection_exists.return_value = False
        mock_target_vector.create_collection.return_value = True
        mock_target_vector.upsert_points.return_value = True

        # Configure factory to return appropriate services
        mock_factory_instance.get_embedding_service.side_effect = [
            mock_source_embedding,
            mock_target_embedding,
        ]
        mock_factory_instance.get_vector_service.side_effect = [
            mock_source_vector,
            mock_target_vector,
        ]

        # Create configurations
        source_config = Config(
            embedding_provider=EmbeddingProvider.OPENAI,
            vector_provider=VectorProvider.QDRANT_CLOUD,
            qdrant_collection="test_collection",
            vector_size=1536,
            openai_api_key="sk-source",
            qdrant_cloud_url="https://source.qdrant.tech",
            qdrant_api_key="source-key",
        )

        target_config = Config(
            embedding_provider=EmbeddingProvider.OPENAI,
            vector_provider=VectorProvider.QDRANT_CLOUD,
            qdrant_collection="test_collection",
            vector_size=1536,
            openai_api_key="sk-test",
            qdrant_cloud_url="https://test.qdrant.tech",
            qdrant_api_key="test-key",
        )

        manager = MigrationManager(source_config, target_config)

        # Test full migration
        report = manager.migrate_vectors(batch_size=2, backup_first=False)

        assert report.total_documents == 2
        assert report.migrated_documents == 2
        assert report.failed_documents == 0
        assert report.success_rate == 100.0

        # Verify migration steps were performed
        mock_target_vector.create_collection.assert_called_once()
        mock_source_embedding.generate_embeddings_batch.assert_not_called()
        mock_target_embedding.generate_embeddings_batch.assert_not_called()
        mock_target_vector.upsert_points.assert_called_once()
