# SERVICES GROUP - Service interfaces and provider management
"""
Multi-provider support for embedding and vector services
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import structlog

logger = structlog.get_logger(__name__)


class EmbeddingProvider(Enum):
    """Supported embedding service providers"""

    OPENAI = "openai"

    def __str__(self) -> str:
        return self.value


class VectorProvider(Enum):
    """Supported vector database providers"""

    QDRANT_CLOUD = "qdrant_cloud"

    def __str__(self) -> str:
        return self.value


class EmbeddingServiceInterface(ABC):
    """Abstract interface for embedding services"""

    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        pass

    @abstractmethod
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the embedding service is healthy"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        pass

    @abstractmethod
    def validate_token_count(self, text: str) -> bool:
        """Validate that text doesn't exceed token limits"""
        pass


class VectorServiceInterface(ABC):
    """Abstract interface for vector database services"""

    @abstractmethod
    def create_collection(self, collection_name: str, vector_size: int) -> bool:
        """Create a new collection"""
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        pass

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        pass

    @abstractmethod
    def upsert_points(self, collection_name: str, points: List[Dict[str, Any]]) -> bool:
        """Insert or update points in the collection"""
        pass

    @abstractmethod
    def delete_points(self, collection_name: str, point_ids: List[str]) -> bool:
        """Delete points from the collection"""
        pass

    @abstractmethod
    def search_points(
        self, collection_name: str, vector: List[float], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for similar points in the collection"""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the vector service is healthy"""
        pass


class ProviderConfig:
    """Configuration for provider selection"""

    def __init__(
        self, embedding_provider: EmbeddingProvider, vector_provider: VectorProvider
    ):
        self.embedding_provider = embedding_provider
        self.vector_provider = vector_provider

    @classmethod
    def from_strings(
        cls, embedding_provider: str, vector_provider: str
    ) -> "ProviderConfig":
        """Create ProviderConfig from string values"""
        try:
            emb_provider = EmbeddingProvider(embedding_provider)
            vec_provider = VectorProvider(vector_provider)
            return cls(emb_provider, vec_provider)
        except ValueError as e:
            raise ValueError(f"Invalid provider string: {e}")

    def is_valid(self) -> bool:
        """Validate the provider configuration"""
        return isinstance(self.embedding_provider, EmbeddingProvider) and isinstance(
            self.vector_provider, VectorProvider
        )

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary representation"""
        return {
            "embedding_provider": str(self.embedding_provider),
            "vector_provider": str(self.vector_provider),
        }


class ProviderFactory:
    """Factory for creating provider service instances"""

    def __init__(self):
        self.logger = logger.bind(component="ProviderFactory")

    def get_embedding_service(
        self, provider: EmbeddingProvider, config: Dict[str, Any]
    ) -> EmbeddingServiceInterface:
        """Get embedding service instance for the specified provider"""
        self.logger.info("Creating embedding service", provider=str(provider))

        if provider == EmbeddingProvider.OPENAI:
            return self._create_openai_embedding_service(config)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def get_vector_service(
        self, provider: VectorProvider, config: Dict[str, Any]
    ) -> VectorServiceInterface:
        """Get vector service instance for the specified provider"""
        self.logger.info("Creating vector service", provider=str(provider))

        if provider == VectorProvider.QDRANT_CLOUD:
            return self._create_qdrant_cloud_service(config)
        else:
            raise ValueError(f"Unsupported vector provider: {provider}")

    def validate_embedding_config(
        self, provider: EmbeddingProvider, config: Dict[str, Any]
    ) -> bool:
        """Validate configuration for embedding provider"""
        if provider == EmbeddingProvider.OPENAI:
            required_fields = ["api_key", "model"]
            return all(field in config for field in required_fields)
        else:
            return False

    def validate_vector_config(
        self, provider: VectorProvider, config: Dict[str, Any]
    ) -> bool:
        """Validate configuration for vector provider"""
        if provider == VectorProvider.QDRANT_CLOUD:
            required_fields = ["url", "api_key"]
            return all(field in config for field in required_fields)
        return False

    def _create_openai_embedding_service(
        self, config: Dict[str, Any]
    ) -> EmbeddingServiceInterface:
        """Create OpenAI embedding service instance"""
        try:
            from .embedding_service_openai import OpenAIEmbeddingService
        except ImportError:
            from embedding_service_openai import OpenAIEmbeddingService
        return OpenAIEmbeddingService(
            api_key=config["api_key"],
            model=config.get("model", "text-embedding-3-small"),
            base_url=config.get("base_url", "https://api.openai.com/v1"),
        )

    def _create_qdrant_cloud_service(
        self, config: Dict[str, Any]
    ) -> VectorServiceInterface:
        """Create Qdrant Cloud service instance"""
        try:
            from .qdrant_service_cloud import QdrantCloudService
        except ImportError:
            from qdrant_service_cloud import QdrantCloudService
        return QdrantCloudService(
            url=config["url"],
            api_key=config["api_key"],
            timeout=config.get("timeout", 30),
            collection_name=config.get("collection_name"),
            vector_size=config.get("vector_size"),
        )


# Convenience functions for backward compatibility
def create_default_provider_config() -> ProviderConfig:
    """Create default provider configuration (OpenAI + Qdrant Cloud)"""
    return ProviderConfig(
        embedding_provider=EmbeddingProvider.OPENAI,
        vector_provider=VectorProvider.QDRANT_CLOUD,
    )


def get_available_embedding_providers() -> List[str]:
    """Get list of available embedding providers"""
    return [str(provider) for provider in EmbeddingProvider]


def get_available_vector_providers() -> List[str]:
    """Get list of available vector providers"""
    return [str(provider) for provider in VectorProvider]
