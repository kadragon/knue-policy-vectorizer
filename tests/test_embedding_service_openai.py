"""
Tests for OpenAI embedding service implementation
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.embedding_service_openai import (
    OpenAIAuthenticationError,
    OpenAIEmbeddingError,
    OpenAIEmbeddingService,
    OpenAIRateLimitError,
)
from src.providers import EmbeddingServiceInterface


class TestOpenAIEmbeddingService:
    """Test OpenAI embedding service implementation"""

    def test_service_implements_interface(self):
        """Test that OpenAIEmbeddingService implements EmbeddingServiceInterface"""
        with patch("src.embedding_service_openai.OpenAI"):
            service = OpenAIEmbeddingService(api_key="test-key")
            assert isinstance(service, EmbeddingServiceInterface)

    def test_service_initialization(self):
        """Test service initialization with different parameters"""
        with patch("src.embedding_service_openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Test with minimal parameters
            service = OpenAIEmbeddingService(api_key="test-key")
            assert service.api_key == "test-key"
            assert service.model == "text-embedding-3-small"
            assert service.max_tokens == 8191
            assert service.embedding_dimension == 1536

            # Test with custom parameters
            service = OpenAIEmbeddingService(
                api_key="custom-key",
                model="text-embedding-3-large",
                base_url="https://custom.openai.com/v1",
                max_tokens=8000,
                timeout=60,
            )
            assert service.api_key == "custom-key"
            assert service.model == "text-embedding-3-large"
            assert service.max_tokens == 8000
            assert (
                service.embedding_dimension == 3072
            )  # text-embedding-3-large dimension

    def test_model_dimension_mapping(self):
        """Test that model dimensions are correctly mapped"""
        with patch("openai.OpenAI"):
            # text-embedding-3-small
            service = OpenAIEmbeddingService(
                api_key="test", model="text-embedding-3-small"
            )
            assert service.embedding_dimension == 1536

            # text-embedding-3-large
            service = OpenAIEmbeddingService(
                api_key="test", model="text-embedding-3-large"
            )
            assert service.embedding_dimension == 3072

            # text-embedding-ada-002 (legacy)
            service = OpenAIEmbeddingService(
                api_key="test", model="text-embedding-ada-002"
            )
            assert service.embedding_dimension == 1536

    def test_generate_embedding_success(self):
        """Test successful embedding generation"""
        with patch("src.embedding_service_openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = [Mock()]
            mock_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4]
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client

            service = OpenAIEmbeddingService(api_key="test-key")
            result = service.generate_embedding("test text")

            assert result == [0.1, 0.2, 0.3, 0.4]
            mock_client.embeddings.create.assert_called_once_with(
                model="text-embedding-3-small",
                input="test text",
                encoding_format="float",
            )

    def test_generate_embedding_with_token_validation(self):
        """Test embedding generation with token validation"""
        with patch("src.embedding_service_openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = [Mock()]
            mock_response.data[0].embedding = [0.1, 0.2, 0.3]
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client

            service = OpenAIEmbeddingService(api_key="test-key", max_tokens=100)

            # Should work with short text
            result = service.generate_embedding("short text")
            assert result == [0.1, 0.2, 0.3]

            # Should fail with very long text
            long_text = "word " * 1000  # This should exceed 100 tokens
            with pytest.raises(
                OpenAIEmbeddingError, match="exceeds maximum token limit"
            ):
                service.generate_embedding(long_text)

    def test_generate_embeddings_batch(self):
        """Test batch embedding generation"""
        with patch("src.embedding_service_openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = [Mock(), Mock(), Mock()]
            mock_response.data[0].embedding = [0.1, 0.2]
            mock_response.data[1].embedding = [0.3, 0.4]
            mock_response.data[2].embedding = [0.5, 0.6]
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client

            service = OpenAIEmbeddingService(api_key="test-key")
            texts = ["text1", "text2", "text3"]
            results = service.generate_embeddings_batch(texts)

            expected = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
            assert results == expected
            mock_client.embeddings.create.assert_called_once_with(
                model="text-embedding-3-small", input=texts, encoding_format="float"
            )

    def test_generate_embeddings_batch_chunking(self):
        """Test batch embedding with automatic chunking for large batches"""
        with patch("src.embedding_service_openai.OpenAI") as mock_openai:
            mock_client = Mock()

            # Mock responses for two chunks
            mock_response1 = Mock()
            mock_response1.data = [Mock(), Mock()]
            mock_response1.data[0].embedding = [0.1, 0.2]
            mock_response1.data[1].embedding = [0.3, 0.4]

            mock_response2 = Mock()
            mock_response2.data = [Mock()]
            mock_response2.data[0].embedding = [0.5, 0.6]

            mock_client.embeddings.create.side_effect = [mock_response1, mock_response2]
            mock_openai.return_value = mock_client

            service = OpenAIEmbeddingService(api_key="test-key", batch_size=2)
            texts = ["text1", "text2", "text3"]
            results = service.generate_embeddings_batch(texts)

            expected = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
            assert results == expected
            assert mock_client.embeddings.create.call_count == 2

    def test_health_check_success(self):
        """Test successful health check"""
        with patch("src.embedding_service_openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = [Mock()]
            # Create embedding with correct dimension for text-embedding-3-small (1536)
            mock_response.data[0].embedding = [0.1] * 1536
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client

            service = OpenAIEmbeddingService(api_key="test-key")
            assert service.health_check() is True

    def test_health_check_failure(self):
        """Test health check failure"""
        with patch("src.embedding_service_openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.embeddings.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client

            service = OpenAIEmbeddingService(api_key="test-key")
            assert service.health_check() is False

    def test_get_model_info(self):
        """Test getting model information"""
        with patch("openai.OpenAI"):
            service = OpenAIEmbeddingService(
                api_key="test-key", model="text-embedding-3-large"
            )

            model_info = service.get_model_info()
            expected = {
                "provider": "openai",
                "model": "text-embedding-3-large",
                "dimension": 3072,
                "max_tokens": 8191,
                "batch_size": 100,
            }
            assert model_info == expected

    def test_validate_token_count(self):
        """Test token count validation"""
        with patch("openai.OpenAI"):
            service = OpenAIEmbeddingService(api_key="test-key", max_tokens=100)

            # Short text should pass
            assert service.validate_token_count("short text") is True

            # Very long text should fail
            long_text = "word " * 1000
            assert service.validate_token_count(long_text) is False

    def test_authentication_error_handling(self):
        """Test handling of authentication errors"""
        with patch("src.embedding_service_openai.OpenAI") as mock_openai:
            mock_client = Mock()

            # Mock OpenAI authentication error
            from openai import AuthenticationError

            # Create a proper exception instance
            mock_response = Mock()
            mock_response.status_code = 401
            auth_error = AuthenticationError(
                message="Invalid API key",
                response=mock_response,
                body={"error": {"message": "Invalid API key"}},
            )
            mock_client.embeddings.create.side_effect = auth_error
            mock_openai.return_value = mock_client

            service = OpenAIEmbeddingService(api_key="invalid-key")

            with pytest.raises(OpenAIAuthenticationError, match="Invalid API key"):
                service.generate_embedding("test")

    def test_rate_limit_error_handling(self):
        """Test handling of rate limit errors"""
        with patch("src.embedding_service_openai.OpenAI") as mock_openai:
            mock_client = Mock()

            # Mock OpenAI rate limit error
            from openai import RateLimitError

            # Create a proper exception instance
            mock_response = Mock()
            mock_response.status_code = 429
            rate_error = RateLimitError(
                message="Rate limit exceeded",
                response=mock_response,
                body={"error": {"message": "Rate limit exceeded"}},
            )
            mock_client.embeddings.create.side_effect = rate_error
            mock_openai.return_value = mock_client

            service = OpenAIEmbeddingService(api_key="test-key")

            with pytest.raises(OpenAIRateLimitError, match="Rate limit exceeded"):
                service.generate_embedding("test")

    def test_generic_openai_error_handling(self):
        """Test handling of generic OpenAI errors"""
        with patch("src.embedding_service_openai.OpenAI") as mock_openai:
            mock_client = Mock()

            # Mock generic OpenAI error
            from openai import OpenAIError

            # Create a proper exception instance
            generic_error = OpenAIError("Generic API error")
            mock_client.embeddings.create.side_effect = generic_error
            mock_openai.return_value = mock_client

            service = OpenAIEmbeddingService(api_key="test-key")

            with pytest.raises(OpenAIEmbeddingError, match="Generic API error"):
                service.generate_embedding("test")

    def test_unexpected_error_handling(self):
        """Test handling of unexpected errors"""
        with patch("src.embedding_service_openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.embeddings.create.side_effect = ValueError("Unexpected error")
            mock_openai.return_value = mock_client

            service = OpenAIEmbeddingService(api_key="test-key")

            with pytest.raises(OpenAIEmbeddingError, match="Unexpected error"):
                service.generate_embedding("test")

    def test_empty_response_handling(self):
        """Test handling of empty responses"""
        with patch("src.embedding_service_openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = []  # Empty response
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client

            service = OpenAIEmbeddingService(api_key="test-key")

            with pytest.raises(OpenAIEmbeddingError, match="No embedding data"):
                service.generate_embedding("test")

    def test_mismatched_batch_response(self):
        """Test handling of mismatched batch responses"""
        with patch("src.embedding_service_openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = [Mock()]  # Only 1 response for 2 inputs
            mock_response.data[0].embedding = [0.1, 0.2]
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client

            service = OpenAIEmbeddingService(api_key="test-key")

            with pytest.raises(OpenAIEmbeddingError, match="Mismatch between input"):
                service.generate_embeddings_batch(["text1", "text2"])

    def test_custom_base_url(self):
        """Test service with custom base URL"""
        with patch("src.embedding_service_openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            service = OpenAIEmbeddingService(
                api_key="test-key", base_url="https://custom.openai.com/v1"
            )

            mock_openai.assert_called_once_with(
                api_key="test-key", base_url="https://custom.openai.com/v1", timeout=30
            )

    def test_batch_size_configuration(self):
        """Test batch size configuration"""
        with patch("openai.OpenAI"):
            service = OpenAIEmbeddingService(api_key="test-key", batch_size=50)
            assert service.batch_size == 50

    def test_timeout_configuration(self):
        """Test timeout configuration"""
        with patch("src.embedding_service_openai.OpenAI") as mock_openai:
            service = OpenAIEmbeddingService(api_key="test-key", timeout=60)

            mock_openai.assert_called_once_with(
                api_key="test-key", base_url="https://api.openai.com/v1", timeout=60
            )
