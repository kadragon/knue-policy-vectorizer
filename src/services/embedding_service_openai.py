"""
OpenAI Embedding Service for KNUE Policy Vectorizer
Integrates with OpenAI API to generate embeddings using text-embedding models
"""

import logging
import time
from typing import Any, Dict, List, Optional

import structlog
import tiktoken
from openai import AuthenticationError, OpenAI, OpenAIError, RateLimitError

try:
    from src.utils.providers import EmbeddingServiceInterface
except ImportError:  # pragma: no cover - fallback for script imports
    from src.utils.providers import EmbeddingServiceInterface

logger = structlog.get_logger(__name__)


class OpenAIEmbeddingError(Exception):
    """Custom exception for OpenAI embedding-related errors"""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        if cause:
            self.__cause__ = cause


class OpenAIAuthenticationError(OpenAIEmbeddingError):
    """Exception for OpenAI authentication errors"""

    pass


class OpenAIRateLimitError(OpenAIEmbeddingError):
    """Exception for OpenAI rate limit errors"""

    pass


class OpenAIEmbeddingService(EmbeddingServiceInterface):
    """
    Service for generating embeddings using OpenAI API

    Provides functionality to:
    - Generate embeddings for single texts or batches
    - Validate token limits before processing
    - Health check for OpenAI API connectivity
    - Model information retrieval
    - Handle rate limiting and authentication errors
    """

    # Model dimension mappings
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,  # Legacy model
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        max_tokens: int = 8191,
        batch_size: int = 100,
        timeout: int = 30,
    ):
        """
        Initialize OpenAI Embedding Service

        Args:
            api_key: OpenAI API key
            model: OpenAI embedding model name
            base_url: Base URL for OpenAI API (allows custom endpoints)
            max_tokens: Maximum tokens allowed per text
            batch_size: Maximum number of texts to process in one batch
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.timeout = timeout

        # Set embedding dimension based on model
        self.embedding_dimension = self.MODEL_DIMENSIONS.get(model, 1536)

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            # Fallback to cl100k_base encoding if model not found
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.logger = logger.bind(
            component="OpenAIEmbeddingService", model=model, max_tokens=max_tokens
        )

        self.logger.info("OpenAI embedding service initialized")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Input text to generate embedding for

        Returns:
            List of floats representing the embedding vector

        Raises:
            OpenAIEmbeddingError: If embedding generation fails
            OpenAIAuthenticationError: If API key is invalid
            OpenAIRateLimitError: If rate limit is exceeded
        """
        if not self.validate_token_count(text):
            raise OpenAIEmbeddingError(
                f"Text with {self._count_tokens(text)} tokens exceeds maximum token limit of {self.max_tokens}"
            )

        try:
            self.logger.debug("Generating embedding", text_length=len(text))

            response = self.client.embeddings.create(
                model=self.model, input=text, encoding_format="float"
            )

            if not response.data:
                raise OpenAIEmbeddingError("No embedding data received from OpenAI API")

            embedding = response.data[0].embedding

            self.logger.debug(
                "Embedding generated successfully", embedding_dimension=len(embedding)
            )

            return embedding

        except AuthenticationError as e:
            raise OpenAIAuthenticationError(f"OpenAI authentication failed: {e}")
        except RateLimitError as e:
            raise OpenAIRateLimitError(f"OpenAI rate limit exceeded: {e}")
        except OpenAIError as e:
            raise OpenAIEmbeddingError(f"OpenAI API error: {e}")
        except Exception as e:
            raise OpenAIEmbeddingError(
                f"Unexpected error during embedding generation: {e}"
            )

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of input texts to generate embeddings for

        Returns:
            List of embedding vectors (each is a list of floats)

        Raises:
            OpenAIEmbeddingError: If embedding generation fails
        """
        if not texts:
            return []

        # Validate all texts
        for i, text in enumerate(texts):
            if not self.validate_token_count(text):
                raise OpenAIEmbeddingError(
                    f"Text at index {i} with {self._count_tokens(text)} tokens exceeds maximum token limit of {self.max_tokens}"
                )

        # Process in chunks if batch is too large
        if len(texts) <= self.batch_size:
            return self._generate_batch_chunk(texts)
        else:
            return self._generate_batch_chunked(texts)

    def _generate_batch_chunk(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a single chunk of texts"""
        try:
            self.logger.debug("Generating batch embeddings", batch_size=len(texts))

            response = self.client.embeddings.create(
                model=self.model, input=texts, encoding_format="float"
            )

            if len(response.data) != len(texts):
                raise OpenAIEmbeddingError(
                    f"Mismatch between input texts ({len(texts)}) and response embeddings ({len(response.data)})"
                )

            embeddings = [item.embedding for item in response.data]

            self.logger.debug(
                "Batch embeddings generated successfully", count=len(embeddings)
            )

            return embeddings

        except AuthenticationError as e:
            raise OpenAIAuthenticationError(f"OpenAI authentication failed: {e}")
        except RateLimitError as e:
            raise OpenAIRateLimitError(f"OpenAI rate limit exceeded: {e}")
        except OpenAIError as e:
            raise OpenAIEmbeddingError(f"OpenAI API error: {e}")
        except Exception as e:
            raise OpenAIEmbeddingError(
                f"Unexpected error during batch embedding generation: {e}"
            )

    def _generate_batch_chunked(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts using chunking for large batches"""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            chunk_embeddings = self._generate_batch_chunk(chunk)
            all_embeddings.extend(chunk_embeddings)

            # Small delay between chunks to be respectful of rate limits
            if i + self.batch_size < len(texts):
                time.sleep(0.1)

        return all_embeddings

    def health_check(self) -> bool:
        """
        Check if the OpenAI API is accessible and working

        Returns:
            True if the service is healthy, False otherwise
        """
        try:
            # Generate a test embedding with a simple text
            test_embedding = self.generate_embedding("test")

            # Verify we got a valid embedding
            is_healthy = (
                isinstance(test_embedding, list)
                and len(test_embedding) == self.embedding_dimension
                and all(isinstance(x, (int, float)) for x in test_embedding)
            )

            self.logger.info("Health check completed", healthy=is_healthy)
            return is_healthy

        except Exception as e:
            self.logger.warning("Health check failed", error=str(e))
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model

        Returns:
            Dictionary containing model information
        """
        return {
            "provider": "openai",
            "model": self.model,
            "dimension": self.embedding_dimension,
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
        }

    def validate_token_count(self, text: str) -> bool:
        """
        Validate that text doesn't exceed token limits

        Args:
            text: Text to validate

        Returns:
            True if text is within token limits, False otherwise
        """
        token_count = self._count_tokens(text)
        return token_count <= self.max_tokens

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens in the text
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # Fallback to rough estimation if tokenizer fails
            # Approximate 4 characters per token
            return len(text) // 4
