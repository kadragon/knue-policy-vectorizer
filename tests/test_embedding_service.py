"""
Test suite for EmbeddingService - Ollama integration
Tests written following TDD approach
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from typing import List, Dict, Any

from src.embedding_service import EmbeddingService, EmbeddingError


class TestEmbeddingService:
    """Test suite for EmbeddingService with Ollama integration"""
    
    @pytest.fixture
    def embedding_service(self) -> EmbeddingService:
        """Create EmbeddingService instance for testing"""
        return EmbeddingService()
    
    @pytest.fixture
    def sample_text(self) -> str:
        """Sample Korean text for testing"""
        return "한국교원대학교 정책 문서입니다. 이 문서는 대학의 운영 방침을 담고 있습니다."
    
    @pytest.fixture
    def sample_long_text(self) -> str:
        """Sample long text that might exceed token limits"""
        return "한국교원대학교 " * 2000  # Approximately 4000 tokens
    
    def test_init_default_config(self, embedding_service):
        """Test EmbeddingService initialization with default configuration"""
        assert embedding_service.model_name == "bge-m3"
        assert embedding_service.base_url == "http://localhost:11434"
        assert embedding_service.embedding_dimension == 1024
        assert embedding_service.max_tokens == 8192
    
    def test_init_custom_config(self):
        """Test EmbeddingService initialization with custom configuration"""
        service = EmbeddingService(
            model_name="custom-model",
            base_url="http://custom:8080",
            embedding_dimension=512,
            max_tokens=4096
        )
        assert service.model_name == "custom-model"
        assert service.base_url == "http://custom:8080"
        assert service.embedding_dimension == 512
        assert service.max_tokens == 4096
    
    @patch('src.embedding_service.OllamaEmbeddings')
    def test_ollama_client_creation(self, mock_ollama, embedding_service):
        """Test that Ollama client is created with correct parameters"""
        # Access the client property to trigger lazy initialization
        client = embedding_service.client
        
        mock_ollama.assert_called_once_with(
            model="bge-m3",
            base_url="http://localhost:11434"
        )
    
    @patch('src.embedding_service.OllamaEmbeddings')
    def test_generate_embedding_success(self, mock_ollama, embedding_service, sample_text):
        """Test successful embedding generation"""
        # Mock the embedding client
        mock_client = Mock()
        mock_embedding = [0.1] * 1024  # 1024-dimensional vector
        mock_client.embed_query.return_value = mock_embedding
        mock_ollama.return_value = mock_client
        
        # Generate embedding
        result = embedding_service.generate_embedding(sample_text)
        
        # Verify results
        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(isinstance(val, float) for val in result)
        mock_client.embed_query.assert_called_once_with(sample_text)
    
    @patch('src.embedding_service.OllamaEmbeddings')
    def test_generate_embedding_ollama_error(self, mock_ollama, embedding_service, sample_text):
        """Test handling of Ollama connection/model errors"""
        # Mock the embedding client to raise an exception
        mock_client = Mock()
        mock_client.embed_query.side_effect = Exception("Ollama connection failed")
        mock_ollama.return_value = mock_client
        
        # Test that EmbeddingError is raised
        with pytest.raises(EmbeddingError, match="Failed to generate embedding"):
            embedding_service.generate_embedding(sample_text)
    
    def test_generate_embedding_empty_text(self, embedding_service):
        """Test handling of empty text input"""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            embedding_service.generate_embedding("")
        
        with pytest.raises(ValueError, match="Text cannot be empty"):
            embedding_service.generate_embedding("   ")  # Only whitespace
    
    def test_generate_embedding_none_text(self, embedding_service):
        """Test handling of None text input"""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            embedding_service.generate_embedding(None)
    
    @patch('src.embedding_service.tiktoken')
    def test_estimate_tokens(self, mock_tiktoken, embedding_service, sample_text):
        """Test token count estimation"""
        # Mock tiktoken encoding
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_tiktoken.get_encoding.return_value = mock_encoding
        
        # Test token estimation
        token_count = embedding_service.estimate_tokens(sample_text)
        
        assert token_count == 5
        mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")
        mock_encoding.encode.assert_called_once_with(sample_text)
    
    @patch('src.embedding_service.tiktoken')
    def test_validate_token_limit_success(self, mock_tiktoken, embedding_service, sample_text):
        """Test successful token limit validation"""
        # Mock tiktoken to return tokens within limit
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1] * 1000  # 1000 tokens (under 8192)
        mock_tiktoken.get_encoding.return_value = mock_encoding
        
        # Should not raise an exception
        embedding_service.validate_token_limit(sample_text)
    
    @patch('src.embedding_service.tiktoken')
    def test_validate_token_limit_exceeded(self, mock_tiktoken, embedding_service, sample_long_text):
        """Test token limit validation when limit is exceeded"""
        # Mock tiktoken to return tokens over limit
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1] * 10000  # 10000 tokens (over 8192)
        mock_tiktoken.get_encoding.return_value = mock_encoding
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Text exceeds maximum token limit"):
            embedding_service.validate_token_limit(sample_long_text)
    
    @patch('src.embedding_service.OllamaEmbeddings')
    def test_generate_embedding_with_validation(self, mock_ollama, embedding_service, sample_text):
        """Test that generate_embedding includes token validation"""
        # Mock the embedding client
        mock_client = Mock()
        mock_embedding = [0.1] * 1024
        mock_client.embed_query.return_value = mock_embedding
        mock_ollama.return_value = mock_client
        
        # Mock tiktoken for validation
        with patch('src.embedding_service.tiktoken') as mock_tiktoken:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1] * 1000  # Under limit
            mock_tiktoken.get_encoding.return_value = mock_encoding
            
            result = embedding_service.generate_embedding(sample_text)
            
            # Verify validation was called
            mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")
            mock_encoding.encode.assert_called_once_with(sample_text)
            
            # Verify embedding generation
            assert len(result) == 1024
    
    @patch('src.embedding_service.OllamaEmbeddings')
    def test_generate_embeddings_batch(self, mock_ollama, embedding_service):
        """Test batch embedding generation"""
        texts = [
            "첫 번째 문서입니다.",
            "두 번째 문서입니다.",
            "세 번째 문서입니다."
        ]
        
        # Mock the embedding client
        mock_client = Mock()
        mock_embeddings = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
        mock_client.embed_documents.return_value = mock_embeddings
        mock_ollama.return_value = mock_client
        
        # Mock tiktoken for validation
        with patch('src.embedding_service.tiktoken') as mock_tiktoken:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1] * 100  # Under limit for each
            mock_tiktoken.get_encoding.return_value = mock_encoding
            
            results = embedding_service.generate_embeddings_batch(texts)
            
            # Verify results
            assert len(results) == 3
            assert all(len(emb) == 1024 for emb in results)
            mock_client.embed_documents.assert_called_once_with(texts)
    
    def test_generate_embeddings_batch_empty_list(self, embedding_service):
        """Test batch embedding with empty list"""
        with pytest.raises(ValueError, match="Text list cannot be empty"):
            embedding_service.generate_embeddings_batch([])
    
    @patch('src.embedding_service.OllamaEmbeddings')
    def test_health_check_success(self, mock_ollama, embedding_service):
        """Test successful health check"""
        # Mock the embedding client
        mock_client = Mock()
        mock_client.embed_query.return_value = [0.1] * 1024
        mock_ollama.return_value = mock_client
        
        # Health check should return True
        assert embedding_service.health_check() is True
        mock_client.embed_query.assert_called_once_with("test")
    
    @patch('src.embedding_service.OllamaEmbeddings')
    def test_health_check_failure(self, mock_ollama, embedding_service):
        """Test health check failure"""
        # Mock the embedding client to fail
        mock_client = Mock()
        mock_client.embed_query.side_effect = Exception("Connection failed")
        mock_ollama.return_value = mock_client
        
        # Health check should return False
        assert embedding_service.health_check() is False
    
    @patch('src.embedding_service.OllamaEmbeddings')
    def test_get_model_info(self, mock_ollama, embedding_service):
        """Test getting model information"""
        info = embedding_service.get_model_info()
        
        expected_info = {
            "model_name": "bge-m3",
            "base_url": "http://localhost:11434",
            "embedding_dimension": 1024,
            "max_tokens": 8192
        }
        
        assert info == expected_info


class TestEmbeddingError:
    """Test EmbeddingError exception class"""
    
    def test_embedding_error_creation(self):
        """Test EmbeddingError exception creation"""
        error_msg = "Test error message"
        error = EmbeddingError(error_msg)
        
        assert str(error) == error_msg
        assert isinstance(error, Exception)
    
    def test_embedding_error_with_cause(self):
        """Test EmbeddingError with original cause"""
        original_error = ValueError("Original error")
        error = EmbeddingError("Wrapped error", original_error)
        
        assert str(error) == "Wrapped error"
        assert error.__cause__ == original_error


# Integration tests (require actual Ollama service)
@pytest.mark.integration
class TestEmbeddingServiceIntegration:
    """Integration tests requiring actual Ollama service"""
    
    @pytest.fixture
    def embedding_service(self) -> EmbeddingService:
        """Create EmbeddingService for integration testing"""
        return EmbeddingService()
    
    def test_actual_ollama_connection(self, embedding_service):
        """Test actual connection to Ollama service"""
        # Skip if Ollama is not available
        if not embedding_service.health_check():
            pytest.skip("Ollama service not available")
        
        # Test with actual Korean text
        text = "한국교원대학교 정책 문서 테스트"
        embedding = embedding_service.generate_embedding(text)
        
        # Verify embedding properties
        assert isinstance(embedding, list)
        assert len(embedding) == 1024
        assert all(isinstance(val, float) for val in embedding)
        assert not all(val == 0 for val in embedding)  # Should not be all zeros
    
    def test_actual_batch_embedding(self, embedding_service):
        """Test actual batch embedding generation"""
        if not embedding_service.health_check():
            pytest.skip("Ollama service not available")
        
        texts = [
            "첫 번째 정책 문서",
            "두 번째 정책 문서",
            "세 번째 정책 문서"
        ]
        
        embeddings = embedding_service.generate_embeddings_batch(texts)
        
        # Verify results
        assert len(embeddings) == 3
        assert all(len(emb) == 1024 for emb in embeddings)
        
        # Embeddings should be different for different texts
        assert embeddings[0] != embeddings[1]
        assert embeddings[1] != embeddings[2]