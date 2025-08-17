"""
Tests for multi-provider support functionality
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest

from src.providers import (
    EmbeddingProvider,
    EmbeddingServiceInterface,
    ProviderConfig,
    ProviderFactory,
    VectorProvider,
    VectorServiceInterface,
)


class TestProviderEnums:
    """Test provider enumeration classes"""

    def test_embedding_provider_enum(self):
        """Test EmbeddingProvider enum has correct values"""
        assert EmbeddingProvider.OLLAMA.value == "ollama"
        assert EmbeddingProvider.OPENAI.value == "openai"

        # Test string conversion
        assert str(EmbeddingProvider.OLLAMA) == "ollama"
        assert str(EmbeddingProvider.OPENAI) == "openai"

    def test_vector_provider_enum(self):
        """Test VectorProvider enum has correct values"""
        assert VectorProvider.QDRANT_LOCAL.value == "qdrant_local"
        assert VectorProvider.QDRANT_CLOUD.value == "qdrant_cloud"

        # Test string conversion
        assert str(VectorProvider.QDRANT_LOCAL) == "qdrant_local"
        assert str(VectorProvider.QDRANT_CLOUD) == "qdrant_cloud"

    def test_embedding_provider_from_string(self):
        """Test creating EmbeddingProvider from string"""
        assert EmbeddingProvider("ollama") == EmbeddingProvider.OLLAMA
        assert EmbeddingProvider("openai") == EmbeddingProvider.OPENAI

        with pytest.raises(ValueError):
            EmbeddingProvider("invalid")

    def test_vector_provider_from_string(self):
        """Test creating VectorProvider from string"""
        assert VectorProvider("qdrant_local") == VectorProvider.QDRANT_LOCAL
        assert VectorProvider("qdrant_cloud") == VectorProvider.QDRANT_CLOUD

        with pytest.raises(ValueError):
            VectorProvider("invalid")


class TestEmbeddingServiceInterface:
    """Test EmbeddingServiceInterface abstract base class"""

    def test_interface_is_abstract(self):
        """Test that EmbeddingServiceInterface cannot be instantiated"""
        with pytest.raises(TypeError):
            EmbeddingServiceInterface()

    def test_interface_methods_are_abstract(self):
        """Test that all required methods are abstract"""

        # Create a mock class that doesn't implement required methods
        class IncompleteEmbeddingService(EmbeddingServiceInterface):
            pass

        with pytest.raises(TypeError):
            IncompleteEmbeddingService()

    def test_concrete_implementation_works(self):
        """Test that concrete implementation of interface works"""

        class ConcreteEmbeddingService(EmbeddingServiceInterface):
            def generate_embedding(self, text: str) -> List[float]:
                return [0.1, 0.2, 0.3]

            def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]

            def health_check(self) -> bool:
                return True

            def get_model_info(self) -> Dict[str, Any]:
                return {"model": "test", "dimension": 3}

            def validate_token_count(self, text: str) -> bool:
                return True

        service = ConcreteEmbeddingService()
        assert service.generate_embedding("test") == [0.1, 0.2, 0.3]
        assert service.health_check() is True


class TestVectorServiceInterface:
    """Test VectorServiceInterface abstract base class"""

    def test_interface_is_abstract(self):
        """Test that VectorServiceInterface cannot be instantiated"""
        with pytest.raises(TypeError):
            VectorServiceInterface()

    def test_concrete_implementation_works(self):
        """Test that concrete implementation of interface works"""

        class ConcreteVectorService(VectorServiceInterface):
            def create_collection(self, collection_name: str, vector_size: int) -> bool:
                return True

            def delete_collection(self, collection_name: str) -> bool:
                return True

            def collection_exists(self, collection_name: str) -> bool:
                return True

            def upsert_points(
                self, collection_name: str, points: List[Dict[str, Any]]
            ) -> bool:
                return True

            def delete_points(self, collection_name: str, point_ids: List[str]) -> bool:
                return True

            def search_points(
                self, collection_name: str, vector: List[float], limit: int = 10
            ) -> List[Dict[str, Any]]:
                return []

            def health_check(self) -> bool:
                return True

        service = ConcreteVectorService()
        assert service.health_check() is True


class TestProviderConfig:
    """Test ProviderConfig class"""

    def test_provider_config_creation(self):
        """Test creating ProviderConfig with different providers"""
        config = ProviderConfig(
            embedding_provider=EmbeddingProvider.OLLAMA,
            vector_provider=VectorProvider.QDRANT_LOCAL,
        )

        assert config.embedding_provider == EmbeddingProvider.OLLAMA
        assert config.vector_provider == VectorProvider.QDRANT_LOCAL

    def test_provider_config_from_strings(self):
        """Test creating ProviderConfig from string values"""
        config = ProviderConfig.from_strings(
            embedding_provider="openai", vector_provider="qdrant_cloud"
        )

        assert config.embedding_provider == EmbeddingProvider.OPENAI
        assert config.vector_provider == VectorProvider.QDRANT_CLOUD

    def test_provider_config_validation(self):
        """Test ProviderConfig validation"""
        # Valid combinations should work
        config = ProviderConfig(
            embedding_provider=EmbeddingProvider.OLLAMA,
            vector_provider=VectorProvider.QDRANT_LOCAL,
        )
        assert config.is_valid()

        # Test with invalid string conversion
        with pytest.raises(ValueError):
            ProviderConfig.from_strings(
                embedding_provider="invalid", vector_provider="qdrant_local"
            )

    def test_provider_config_to_dict(self):
        """Test converting ProviderConfig to dictionary"""
        config = ProviderConfig(
            embedding_provider=EmbeddingProvider.OPENAI,
            vector_provider=VectorProvider.QDRANT_CLOUD,
        )

        config_dict = config.to_dict()
        expected = {"embedding_provider": "openai", "vector_provider": "qdrant_cloud"}
        assert config_dict == expected


class TestProviderFactory:
    """Test ProviderFactory class"""

    def test_factory_creation(self):
        """Test creating ProviderFactory"""
        factory = ProviderFactory()
        assert factory is not None

    def test_get_embedding_service_ollama(self):
        """Test getting Ollama embedding service from factory"""
        factory = ProviderFactory()

        with patch("src.embedding_service.EmbeddingService") as mock_service:
            mock_instance = Mock()
            mock_service.return_value = mock_instance

            service = factory.get_embedding_service(
                EmbeddingProvider.OLLAMA,
                {"ollama_url": "http://localhost:11434", "model": "bge-m3"},
            )

            assert service == mock_instance
            mock_service.assert_called_once()

    def test_get_embedding_service_openai(self):
        """Test getting OpenAI embedding service from factory"""
        factory = ProviderFactory()

        # Mock the import and class creation
        with patch.object(factory, "_create_openai_embedding_service") as mock_create:
            mock_instance = Mock()
            mock_create.return_value = mock_instance

            service = factory.get_embedding_service(
                EmbeddingProvider.OPENAI,
                {"api_key": "test-key", "model": "text-embedding-3-small"},
            )

            assert service == mock_instance
            mock_create.assert_called_once_with(
                {"api_key": "test-key", "model": "text-embedding-3-small"}
            )

    def test_get_vector_service_qdrant_local(self):
        """Test getting local Qdrant service from factory"""
        factory = ProviderFactory()

        with patch("src.qdrant_service.QdrantService") as mock_service:
            mock_instance = Mock()
            mock_service.return_value = mock_instance

            service = factory.get_vector_service(
                VectorProvider.QDRANT_LOCAL, {"url": "http://localhost:6333"}
            )

            assert service == mock_instance
            mock_service.assert_called_once()

    def test_get_vector_service_qdrant_cloud(self):
        """Test getting Qdrant Cloud service from factory"""
        factory = ProviderFactory()

        # Mock the import and class creation
        with patch.object(factory, "_create_qdrant_cloud_service") as mock_create:
            mock_instance = Mock()
            mock_create.return_value = mock_instance

            service = factory.get_vector_service(
                VectorProvider.QDRANT_CLOUD,
                {"url": "https://test.qdrant.tech", "api_key": "test-key"},
            )

            assert service == mock_instance
            mock_create.assert_called_once_with(
                {"url": "https://test.qdrant.tech", "api_key": "test-key"}
            )

    def test_unsupported_embedding_provider(self):
        """Test error handling for unsupported embedding provider"""
        factory = ProviderFactory()

        # Create a mock enum value that's not supported
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            # This will fail when we try to add new providers but don't update the factory
            fake_provider = Mock()
            fake_provider.value = "unsupported"
            factory.get_embedding_service(fake_provider, {})

    def test_unsupported_vector_provider(self):
        """Test error handling for unsupported vector provider"""
        factory = ProviderFactory()

        # Create a mock enum value that's not supported
        with pytest.raises(ValueError, match="Unsupported vector provider"):
            fake_provider = Mock()
            fake_provider.value = "unsupported"
            factory.get_vector_service(fake_provider, {})


class TestProviderConfigValidation:
    """Test provider configuration validation"""

    def test_ollama_config_validation(self):
        """Test Ollama configuration validation"""
        factory = ProviderFactory()

        # Valid config
        valid_config = {"ollama_url": "http://localhost:11434", "model": "bge-m3"}
        assert factory.validate_embedding_config(EmbeddingProvider.OLLAMA, valid_config)

        # Missing required field
        invalid_config = {"ollama_url": "http://localhost:11434"}
        assert not factory.validate_embedding_config(
            EmbeddingProvider.OLLAMA, invalid_config
        )

    def test_openai_config_validation(self):
        """Test OpenAI configuration validation"""
        factory = ProviderFactory()

        # Valid config
        valid_config = {"api_key": "sk-test", "model": "text-embedding-3-small"}
        assert factory.validate_embedding_config(EmbeddingProvider.OPENAI, valid_config)

        # Missing API key
        invalid_config = {"model": "text-embedding-3-small"}
        assert not factory.validate_embedding_config(
            EmbeddingProvider.OPENAI, invalid_config
        )

    def test_qdrant_local_config_validation(self):
        """Test Qdrant local configuration validation"""
        factory = ProviderFactory()

        # Valid config
        valid_config = {"url": "http://localhost:6333"}
        assert factory.validate_vector_config(VectorProvider.QDRANT_LOCAL, valid_config)

        # Missing URL
        invalid_config = {}
        assert not factory.validate_vector_config(
            VectorProvider.QDRANT_LOCAL, invalid_config
        )

    def test_qdrant_cloud_config_validation(self):
        """Test Qdrant Cloud configuration validation"""
        factory = ProviderFactory()

        # Valid config
        valid_config = {"url": "https://test.qdrant.tech", "api_key": "test-key"}
        assert factory.validate_vector_config(VectorProvider.QDRANT_CLOUD, valid_config)

        # Missing API key
        invalid_config = {"url": "https://test.qdrant.tech"}
        assert not factory.validate_vector_config(
            VectorProvider.QDRANT_CLOUD, invalid_config
        )
