"""
Tests for multi-provider configuration support
"""

import os
import pytest
from unittest.mock import patch

from src.config import Config
from src.providers import EmbeddingProvider, VectorProvider, ProviderConfig, ProviderFactory


class TestMultiProviderConfig:
    """Test multi-provider configuration functionality"""

    def test_default_provider_configuration(self):
        """Test default provider configuration (backward compatibility)"""
        config = Config()
        
        # Should default to existing providers
        assert config.embedding_provider == EmbeddingProvider.OLLAMA
        assert config.vector_provider == VectorProvider.QDRANT_LOCAL

    def test_provider_configuration_from_env(self):
        """Test provider configuration from environment variables"""
        with patch.dict(os.environ, {
            'EMBEDDING_PROVIDER': 'openai',
            'VECTOR_PROVIDER': 'qdrant_cloud'
        }):
            config = Config.from_env()
            
            assert config.embedding_provider == EmbeddingProvider.OPENAI
            assert config.vector_provider == VectorProvider.QDRANT_CLOUD

    def test_provider_configuration_validation(self):
        """Test provider configuration validation"""
        with patch.dict(os.environ, {
            'EMBEDDING_PROVIDER': 'invalid_provider',
            'VECTOR_PROVIDER': 'qdrant_local'
        }):
            with pytest.raises(ValueError, match="Invalid embedding provider"):
                Config.from_env()

    def test_openai_configuration_from_env(self):
        """Test OpenAI-specific configuration from environment variables"""
        with patch.dict(os.environ, {
            'EMBEDDING_PROVIDER': 'openai',
            'OPENAI_API_KEY': 'sk-test-key',
            'OPENAI_MODEL': 'text-embedding-3-large',
            'OPENAI_BASE_URL': 'https://custom.openai.com/v1'
        }):
            config = Config.from_env()
            
            assert config.openai_api_key == 'sk-test-key'
            assert config.openai_model == 'text-embedding-3-large'
            assert config.openai_base_url == 'https://custom.openai.com/v1'

    def test_qdrant_cloud_configuration_from_env(self):
        """Test Qdrant Cloud-specific configuration from environment variables"""
        with patch.dict(os.environ, {
            'VECTOR_PROVIDER': 'qdrant_cloud',
            'QDRANT_CLOUD_URL': 'https://test.qdrant.tech',
            'QDRANT_API_KEY': 'test-api-key',
            'QDRANT_CLUSTER_REGION': 'us-east-1'
        }):
            config = Config.from_env()
            
            assert config.qdrant_cloud_url == 'https://test.qdrant.tech'
            assert config.qdrant_api_key == 'test-api-key'
            assert config.qdrant_cluster_region == 'us-east-1'

    def test_provider_config_creation(self):
        """Test creating ProviderConfig from Config"""
        config = Config(
            embedding_provider=EmbeddingProvider.OPENAI,
            vector_provider=VectorProvider.QDRANT_CLOUD
        )
        
        provider_config = config.get_provider_config()
        assert provider_config.embedding_provider == EmbeddingProvider.OPENAI
        assert provider_config.vector_provider == VectorProvider.QDRANT_CLOUD

    def test_embedding_service_config_ollama(self):
        """Test getting embedding service configuration for Ollama"""
        config = Config(
            embedding_provider=EmbeddingProvider.OLLAMA,
            ollama_url="http://localhost:11434",
            embedding_model="bge-m3",
            max_tokens=8192
        )
        
        service_config = config.get_embedding_service_config()
        expected = {
            "ollama_url": "http://localhost:11434",
            "model": "bge-m3",
            "max_tokens": 8192
        }
        assert service_config == expected

    def test_embedding_service_config_openai(self):
        """Test getting embedding service configuration for OpenAI"""
        config = Config(
            embedding_provider=EmbeddingProvider.OPENAI,
            openai_api_key="sk-test",
            openai_model="text-embedding-3-small",
            openai_base_url="https://api.openai.com/v1",
            max_tokens=8191
        )
        
        service_config = config.get_embedding_service_config()
        expected = {
            "api_key": "sk-test",
            "model": "text-embedding-3-small",
            "base_url": "https://api.openai.com/v1",
            "max_tokens": 8191
        }
        assert service_config == expected

    def test_vector_service_config_qdrant_local(self):
        """Test getting vector service configuration for local Qdrant"""
        config = Config(
            vector_provider=VectorProvider.QDRANT_LOCAL,
            qdrant_url="http://localhost:6333"
        )
        
        service_config = config.get_vector_service_config()
        expected = {
            "url": "http://localhost:6333"
        }
        assert service_config == expected

    def test_vector_service_config_qdrant_cloud(self):
        """Test getting vector service configuration for Qdrant Cloud"""
        config = Config(
            vector_provider=VectorProvider.QDRANT_CLOUD,
            qdrant_cloud_url="https://test.qdrant.tech",
            qdrant_api_key="test-key"
        )
        
        service_config = config.get_vector_service_config()
        expected = {
            "url": "https://test.qdrant.tech",
            "api_key": "test-key"
        }
        assert service_config == expected

    def test_config_validation_openai_missing_api_key(self):
        """Test validation when OpenAI provider is selected but API key is missing"""
        config = Config(
            embedding_provider=EmbeddingProvider.OPENAI,
            openai_api_key=""  # Missing API key
        )
        
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            config.validate()

    def test_config_validation_qdrant_cloud_missing_api_key(self):
        """Test validation when Qdrant Cloud provider is selected but API key is missing"""
        config = Config(
            vector_provider=VectorProvider.QDRANT_CLOUD,
            qdrant_api_key=""  # Missing API key
        )
        
        with pytest.raises(ValueError, match="Qdrant Cloud API key is required"):
            config.validate()

    def test_config_validation_qdrant_cloud_missing_url(self):
        """Test validation when Qdrant Cloud provider is selected but URL is missing"""
        config = Config(
            vector_provider=VectorProvider.QDRANT_CLOUD,
            qdrant_cloud_url="",  # Missing URL
            qdrant_api_key="test-key"
        )
        
        with pytest.raises(ValueError, match="Qdrant Cloud URL is required"):
            config.validate()

    def test_provider_factory_creation_from_config(self):
        """Test creating services using ProviderFactory from config"""
        config = Config(
            embedding_provider=EmbeddingProvider.OLLAMA,
            vector_provider=VectorProvider.QDRANT_LOCAL,
            ollama_url="http://localhost:11434",
            embedding_model="bge-m3",
            qdrant_url="http://localhost:6333"
        )
        
        factory = ProviderFactory()
        
        # Test creating services from config
        embedding_config = config.get_embedding_service_config()
        vector_config = config.get_vector_service_config()
        
        # Validate configurations
        assert factory.validate_embedding_config(config.embedding_provider, embedding_config)
        assert factory.validate_vector_config(config.vector_provider, vector_config)

    def test_backward_compatibility_env_vars(self):
        """Test that existing environment variables still work"""
        with patch.dict(os.environ, {
            # Explicitly set providers to ensure isolation and clarity
            'EMBEDDING_PROVIDER': 'ollama',
            'VECTOR_PROVIDER': 'qdrant_local',
            # Legacy-compatible vars should still be respected
            'OLLAMA_URL': 'http://custom:11434',
            'OLLAMA_MODEL': 'custom-model',
            'QDRANT_URL': 'http://custom:6333'
        }):
            config = Config.from_env()
            
            # Should still use existing defaults for providers
            assert config.embedding_provider == EmbeddingProvider.OLLAMA
            assert config.vector_provider == VectorProvider.QDRANT_LOCAL
            
            # But pick up the custom URLs
            assert config.ollama_url == 'http://custom:11434'
            assert config.embedding_model == 'custom-model'
            assert config.qdrant_url == 'http://custom:6333'

    def test_mixed_provider_environment_config(self):
        """Test configuration with mixed providers and their respective settings"""
        with patch.dict(os.environ, {
            'EMBEDDING_PROVIDER': 'openai',
            'VECTOR_PROVIDER': 'qdrant_cloud',
            'OPENAI_API_KEY': 'sk-test',
            'OPENAI_MODEL': 'text-embedding-3-large',
            'QDRANT_CLOUD_URL': 'https://test.qdrant.tech',
            'QDRANT_API_KEY': 'cloud-key'
        }):
            config = Config.from_env()
            
            assert config.embedding_provider == EmbeddingProvider.OPENAI
            assert config.vector_provider == VectorProvider.QDRANT_CLOUD
            assert config.openai_api_key == 'sk-test'
            assert config.openai_model == 'text-embedding-3-large'
            assert config.qdrant_cloud_url == 'https://test.qdrant.tech'
            assert config.qdrant_api_key == 'cloud-key'

    def test_config_dict_export(self):
        """Test exporting configuration as dictionary"""
        config = Config(
            embedding_provider=EmbeddingProvider.OPENAI,
            vector_provider=VectorProvider.QDRANT_CLOUD,
            openai_api_key="sk-test",
            qdrant_cloud_url="https://test.qdrant.tech"
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["embedding_provider"] == "openai"
        assert config_dict["vector_provider"] == "qdrant_cloud"
        assert "openai_api_key" in config_dict
        assert "qdrant_cloud_url" in config_dict

    def test_config_from_dict(self):
        """Test creating configuration from dictionary"""
        config_dict = {
            "embedding_provider": "openai",
            "vector_provider": "qdrant_cloud",
            "openai_api_key": "sk-test",
            "openai_model": "text-embedding-3-small",
            "qdrant_cloud_url": "https://test.qdrant.tech",
            "qdrant_api_key": "test-key"
        }
        
        config = Config.from_dict(config_dict)
        
        assert config.embedding_provider == EmbeddingProvider.OPENAI
        assert config.vector_provider == VectorProvider.QDRANT_CLOUD
        assert config.openai_api_key == "sk-test"
        assert config.openai_model == "text-embedding-3-small"
