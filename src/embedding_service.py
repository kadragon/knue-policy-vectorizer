"""
Embedding Service for KNUE Policy Vectorizer
Integrates with Ollama to generate embeddings using bge-m3 model
"""

import logging
from typing import Any, Dict, List, Optional

import tiktoken
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors"""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        if cause:
            self.__cause__ = cause


class EmbeddingService:
    """
    Service for generating embeddings using Ollama with bge-m3 model

    Provides functionality to:
    - Generate embeddings for single texts or batches
    - Validate token limits before processing
    - Health check for Ollama connectivity
    - Model information retrieval
    """

    def __init__(
        self,
        model_name: str = "bge-m3",
        base_url: str = "http://localhost:11434",
        embedding_dimension: int = 1024,
        max_tokens: int = 8192,
    ):
        """
        Initialize EmbeddingService

        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
            embedding_dimension: Expected dimension of embeddings
            max_tokens: Maximum tokens allowed per text
        """
        self.model_name = model_name
        self.base_url = base_url
        self.embedding_dimension = embedding_dimension
        self.max_tokens = max_tokens

        # Lazy initialization of client
        self._client: Optional[OllamaEmbeddings] = None

        logger.info(
            f"EmbeddingService initialized: model={model_name}, "
            f"url={base_url}, dim={embedding_dimension}, max_tokens={max_tokens}"
        )

    @property
    def client(self) -> OllamaEmbeddings:
        """
        Lazy initialization of Ollama client

        Returns:
            OllamaEmbeddings client instance
        """
        if self._client is None:
            self._client = OllamaEmbeddings(
                model=self.model_name, base_url=self.base_url
            )
            logger.debug(f"Created Ollama client: {self.model_name}@{self.base_url}")

        return self._client

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate number of tokens in text using tiktoken

        Args:
            text: Input text to analyze

        Returns:
            Estimated token count
        """
        if not text or not text.strip():
            return 0

        try:
            # Use OpenAI's tokenizer (cl100k_base) as approximation
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Failed to estimate tokens, using character count: {e}")
            # Fallback to rough character-based estimation (4 chars per token)
            return len(text) // 4

    def validate_token_limit(self, text: str) -> None:
        """
        Validate that text doesn't exceed token limits

        Args:
            text: Input text to validate

        Raises:
            ValueError: If text exceeds maximum token limit
        """
        token_count = self.estimate_tokens(text)
        if token_count > self.max_tokens:
            raise ValueError(
                f"Text exceeds maximum token limit ({token_count} > {self.max_tokens}). "
                f"Please truncate the text or split it into smaller chunks."
            )

        logger.debug(f"Token validation passed: {token_count}/{self.max_tokens}")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Input text to embed

        Returns:
            List of float values representing the embedding vector

        Raises:
            ValueError: If text is empty or exceeds token limits
            EmbeddingError: If embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or whitespace only")

        # Validate token limits
        self.validate_token_limit(text)

        try:
            logger.debug(f"Generating embedding for text (length: {len(text)})")
            embedding = self.client.embed_query(text)

            if not embedding:
                raise EmbeddingError("Generated embedding is empty")

            if len(embedding) != self.embedding_dimension:
                logger.warning(
                    f"Unexpected embedding dimension: {len(embedding)} "
                    f"(expected {self.embedding_dimension})"
                )

            logger.debug(f"Generated embedding with dimension: {len(embedding)}")
            return embedding

        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise

            error_msg = f"Failed to generate embedding: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, e)

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If text list is empty or any text exceeds limits
            EmbeddingError: If batch embedding generation fails
        """
        if not texts:
            raise ValueError("Text list cannot be empty")

        # Validate all texts
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} is empty or whitespace only")
            self.validate_token_limit(text)

        try:
            logger.debug(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.client.embed_documents(texts)

            if not embeddings or len(embeddings) != len(texts):
                raise EmbeddingError(
                    f"Expected {len(texts)} embeddings, got {len(embeddings) if embeddings else 0}"
                )

            # Validate embedding dimensions
            for i, embedding in enumerate(embeddings):
                if len(embedding) != self.embedding_dimension:
                    logger.warning(
                        f"Embedding {i} has unexpected dimension: {len(embedding)} "
                        f"(expected {self.embedding_dimension})"
                    )

            logger.debug(f"Generated {len(embeddings)} embeddings successfully")
            return embeddings

        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise

            error_msg = f"Failed to generate batch embeddings: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, e)

    def health_check(self) -> bool:
        """
        Check if the Ollama service is healthy and responsive

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            logger.debug("Performing health check...")
            test_embedding = self.client.embed_query("test")

            if not test_embedding or len(test_embedding) == 0:
                logger.warning("Health check failed: empty embedding returned")
                return False

            logger.debug("Health check passed")
            return True

        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model configuration information

        Returns:
            Dictionary containing model configuration
        """
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "embedding_dimension": self.embedding_dimension,
            "max_tokens": self.max_tokens,
        }

    def __str__(self) -> str:
        """String representation of the service"""
        return (
            f"EmbeddingService(model={self.model_name}, "
            f"url={self.base_url}, dim={self.embedding_dimension})"
        )

    def __repr__(self) -> str:
        """Detailed representation of the service"""
        return (
            f"EmbeddingService(model_name='{self.model_name}', "
            f"base_url='{self.base_url}', embedding_dimension={self.embedding_dimension}, "
            f"max_tokens={self.max_tokens})"
        )
