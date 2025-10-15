"""
Qdrant Cloud Service for KNUE Policy Vectorizer
Integrates with Qdrant Cloud using API key authentication
"""

import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    UpdateStatus,
    VectorParams,
)

from src.utils.providers import VectorServiceInterface

logger = structlog.get_logger(__name__)


class QdrantCloudError(Exception):
    """Custom exception for Qdrant Cloud-related errors"""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        if cause:
            self.__cause__ = cause


class QdrantCloudAuthenticationError(QdrantCloudError):
    """Exception for Qdrant Cloud authentication errors"""

    pass


class QdrantCloudConnectionError(QdrantCloudError):
    """Exception for Qdrant Cloud connection errors"""

    pass


class QdrantCloudService(VectorServiceInterface):
    """
    Service for managing vectors in Qdrant Cloud

    Provides functionality to:
    - Manage collections (create, delete, check existence)
    - Insert, update, and delete points with vectors
    - Search for similar vectors
    - Health check and connection management
    - Handle cloud-specific authentication and security
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        timeout: int = 30,
        batch_size: int = 100,
        prefer_grpc: bool = False,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
    ):
        """
        Initialize Qdrant Cloud Service

        Args:
            url: Qdrant Cloud cluster URL (must be HTTPS)
            api_key: Qdrant Cloud API key for authentication
            timeout: Request timeout in seconds
            batch_size: Maximum number of points to process in one batch
            prefer_grpc: Whether to prefer gRPC over HTTP
        """
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = batch_size
        self.prefer_grpc = prefer_grpc
        # Optional default collection context for legacy API compatibility
        self.collection_name = collection_name
        self.vector_size = vector_size

        # Validate inputs
        self._validate_configuration()

        # Initialize Qdrant client with cloud authentication
        self.client = QdrantClient(
            url=url, api_key=api_key, timeout=timeout, prefer_grpc=prefer_grpc
        )

        self.logger = logger.bind(
            component="QdrantCloudService", url=url, timeout=timeout
        )

        self.logger.info("Qdrant Cloud service initialized")

    def _validate_configuration(self) -> None:
        """Validate Qdrant Cloud configuration"""
        if not self.api_key or not self.api_key.strip():
            raise ValueError("API key is required for Qdrant Cloud")

        # Parse URL to validate format
        parsed_url = urlparse(self.url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid Qdrant Cloud URL: {self.url}")

        # Qdrant Cloud requires HTTPS for security
        if parsed_url.scheme != "https":
            hostname = parsed_url.hostname
            if hostname and (
                hostname.endswith(".qdrant.tech")
                or hostname == "qdrant.tech"
                or hostname.endswith(".qdrant.io")
                or hostname == "qdrant.io"
            ):
                raise ValueError("Qdrant Cloud requires HTTPS URLs")
            else:
                # Allow HTTP for custom cloud deployments but warn
                self.logger.warning("Using HTTP with Qdrant Cloud may be insecure")

    # ---- New API (explicit arguments) ----
    def create_collection(
        self, collection_name: Optional[str] = None, vector_size: Optional[int] = None
    ) -> bool:
        """
        Create a new collection in Qdrant Cloud

        Args:
            collection_name: Name of the collection to create
            vector_size: Dimension of vectors to store

        Returns:
            True if collection was created successfully

        Raises:
            QdrantCloudError: If collection creation fails
        """
        # Allow using stored defaults if not provided (legacy compatibility)
        if collection_name is None:
            if self.collection_name is None:
                raise QdrantCloudError(
                    "collection_name must be provided or configured on service"
                )
            collection_name = self.collection_name
        if vector_size is None:
            if self.vector_size is None:
                raise QdrantCloudError(
                    "vector_size must be provided or configured on service"
                )
            vector_size = self.vector_size

        try:
            self.logger.info(
                "Creating collection",
                collection_name=collection_name,
                vector_size=vector_size,
            )

            vectors_config = VectorParams(size=vector_size, distance=Distance.COSINE)

            self.client.create_collection(
                collection_name=collection_name, vectors_config=vectors_config
            )

            self.logger.info(
                "Collection created successfully", collection_name=collection_name
            )
            return True

        except UnexpectedResponse as e:
            if e.status_code == 401:
                raise QdrantCloudAuthenticationError(
                    "Authentication failed: Invalid API key"
                )
            elif e.status_code == 400:
                raise QdrantCloudError(f"Bad request: {e.content}")
            else:
                raise QdrantCloudError(f"Qdrant Cloud operation failed: {e}")
        except ResponseHandlingException as e:
            raise QdrantCloudConnectionError(f"Connection failed: {e}")
        except Exception as e:
            raise QdrantCloudError(f"Unexpected error during collection creation: {e}")

    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        Delete a collection from Qdrant Cloud

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if collection was deleted successfully

        Raises:
            QdrantCloudError: If collection deletion fails
        """
        if collection_name is None:
            if self.collection_name is None:
                raise QdrantCloudError(
                    "collection_name must be provided or configured on service"
                )
            collection_name = self.collection_name

        try:
            self.logger.info("Deleting collection", collection_name=collection_name)

            self.client.delete_collection(collection_name)

            self.logger.info(
                "Collection deleted successfully", collection_name=collection_name
            )
            return True

        except UnexpectedResponse as e:
            if e.status_code == 401:
                raise QdrantCloudAuthenticationError(
                    "Authentication failed: Invalid API key"
                )
            elif e.status_code == 404:
                # Collection doesn't exist - consider this success
                self.logger.info(
                    "Collection does not exist", collection_name=collection_name
                )
                return True
            else:
                raise QdrantCloudError(f"Qdrant Cloud operation failed: {e}")
        except ResponseHandlingException as e:
            raise QdrantCloudConnectionError(f"Connection failed: {e}")
        except Exception as e:
            raise QdrantCloudError(f"Unexpected error during collection deletion: {e}")

    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """
        Check if a collection exists in Qdrant Cloud

        Args:
            collection_name: Name of the collection to check

        Returns:
            True if collection exists, False otherwise
        """
        if collection_name is None:
            if self.collection_name is None:
                raise QdrantCloudError(
                    "collection_name must be provided or configured on service"
                )
            collection_name = self.collection_name

        try:
            self.client.get_collection(collection_name)
            return True
        except UnexpectedResponse as e:
            if e.status_code == 404:
                return False
            else:
                # Re-raise other errors (like auth errors)
                raise QdrantCloudError(f"Error checking collection existence: {e}")
        except Exception as e:
            raise QdrantCloudError(f"Unexpected error checking collection: {e}")

    def upsert_points(self, collection_name: str, points: List[Dict[str, Any]]) -> bool:
        """
        Insert or update points in a collection

        Args:
            collection_name: Name of the collection
            points: List of points to upsert, each containing id, vector, and payload

        Returns:
            True if points were upserted successfully

        Raises:
            QdrantCloudError: If upsert operation fails
        """
        if not points:
            return True

        # Validate point structure
        for i, point in enumerate(points):
            if "id" not in point or "vector" not in point:
                raise QdrantCloudError(
                    f"Invalid point structure at index {i}: missing 'id' or 'vector'"
                )

        try:
            self.logger.info(
                "Upserting points",
                collection_name=collection_name,
                points_count=len(points),
            )

            # Process in batches to handle large uploads
            if len(points) <= self.batch_size:
                return self._upsert_batch(collection_name, points)
            else:
                return self._upsert_chunked(collection_name, points)

        except Exception as e:
            if isinstance(
                e,
                (
                    QdrantCloudError,
                    QdrantCloudAuthenticationError,
                    QdrantCloudConnectionError,
                ),
            ):
                raise
            else:
                raise QdrantCloudError(f"Unexpected error during point upsert: {e}")

    def _upsert_batch(self, collection_name: str, points: List[Dict[str, Any]]) -> bool:
        """Upsert a single batch of points"""
        try:
            # Convert to Qdrant PointStruct format
            qdrant_points = []
            for point in points:
                qdrant_point = PointStruct(
                    id=point["id"],
                    vector=point["vector"],
                    payload=point.get("payload", {}),
                )
                qdrant_points.append(qdrant_point)

            self.client.upsert(collection_name=collection_name, points=qdrant_points)

            self.logger.debug(
                "Batch upserted successfully",
                collection_name=collection_name,
                batch_size=len(points),
            )
            return True

        except UnexpectedResponse as e:
            if e.status_code == 401:
                raise QdrantCloudAuthenticationError(
                    "Authentication failed: Invalid API key"
                )
            else:
                raise QdrantCloudError(f"Qdrant Cloud operation failed: {e}")
        except ResponseHandlingException as e:
            raise QdrantCloudConnectionError(f"Connection failed: {e}")

    def _upsert_chunked(
        self, collection_name: str, points: List[Dict[str, Any]]
    ) -> bool:
        """Upsert points using chunking for large batches"""
        for i in range(0, len(points), self.batch_size):
            chunk = points[i : i + self.batch_size]
            self._upsert_batch(collection_name, chunk)

            # Small delay between chunks to be respectful of rate limits
            if i + self.batch_size < len(points):
                time.sleep(0.1)

        self.logger.info(
            "All chunks upserted successfully",
            collection_name=collection_name,
            total_points=len(points),
        )
        return True

    def delete_points(self, collection_name: str, point_ids: List[str]) -> bool:
        """
        Delete points from a collection

        Args:
            collection_name: Name of the collection
            point_ids: List of point IDs to delete

        Returns:
            True if points were deleted successfully

        Raises:
            QdrantCloudError: If delete operation fails
        """
        if not point_ids:
            return True

        try:
            self.logger.info(
                "Deleting points",
                collection_name=collection_name,
                points_count=len(point_ids),
            )

            self.client.delete(
                collection_name=collection_name, points_selector=point_ids
            )

            self.logger.info(
                "Points deleted successfully",
                collection_name=collection_name,
                deleted_count=len(point_ids),
            )
            return True

        except UnexpectedResponse as e:
            if e.status_code == 401:
                raise QdrantCloudAuthenticationError(
                    "Authentication failed: Invalid API key"
                )
            else:
                raise QdrantCloudError(f"Qdrant Cloud operation failed: {e}")
        except ResponseHandlingException as e:
            raise QdrantCloudConnectionError(f"Connection failed: {e}")
        except Exception as e:
            raise QdrantCloudError(f"Unexpected error during point deletion: {e}")

    # ---- Legacy API (used by SyncPipeline) ----
    def upsert_point(
        self, point_id: str, vector: List[float], metadata: Dict[str, Any]
    ) -> bool:
        """Upsert a single point using configured collection."""
        if self.collection_name is None:
            raise QdrantCloudError(
                "collection_name must be configured to use upsert_point"
            )

        try:
            qdrant_point = PointStruct(id=point_id, vector=vector, payload=metadata)
            self.client.upsert(
                collection_name=self.collection_name, points=[qdrant_point]
            )
            return True
        except UnexpectedResponse as e:
            if e.status_code == 401:
                raise QdrantCloudAuthenticationError(
                    "Authentication failed: Invalid API key"
                )
            else:
                raise QdrantCloudError(f"Qdrant Cloud operation failed: {e}")
        except ResponseHandlingException as e:
            raise QdrantCloudConnectionError(f"Connection failed: {e}")

    def upsert_points_batch(self, points_data: List[Dict[str, Any]]) -> bool:
        """Upsert multiple points using configured collection."""
        if self.collection_name is None:
            raise QdrantCloudError(
                "collection_name must be configured to use upsert_points_batch"
            )

        if not points_data:
            return True

        # Map to generic upsert_points structure
        points: List[Dict[str, Any]] = []
        for item in points_data:
            points.append(
                {
                    "id": item["point_id"],
                    "vector": item["vector"],
                    "payload": item["metadata"],
                }
            )
        return self.upsert_points(self.collection_name, points)

    def delete_document_chunks(self, base_document_id: str) -> bool:
        """Delete all points whose payload has document_id equal to the base ID."""
        if self.collection_name is None:
            raise QdrantCloudError(
                "collection_name must be configured to use delete_document_chunks"
            )

        try:
            point_ids: List[str] = []
            scroll_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id", match=MatchValue(value=base_document_id)
                    )
                ]
            )

            offset = None
            limit = 100
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=scroll_filter,
                    limit=limit,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                if not points:
                    break
                point_ids.extend([p.id for p in points])
                if offset is None:
                    break

            if not point_ids:
                self.logger.info(
                    "No chunks found for document", base_document_id=base_document_id
                )
                return True

            result = self.client.delete(
                collection_name=self.collection_name, points_selector=point_ids
            )
            # result may not carry UpdateStatus in HTTP mode; consider success if no exception
            if hasattr(result, "status"):
                return result.status == UpdateStatus.COMPLETED

            self.logger.debug(
                "Delete operation result did not have a status attribute, assuming success.",
                result=result,
            )
            return True
        except UnexpectedResponse as e:
            if e.status_code == 401:
                raise QdrantCloudAuthenticationError(
                    "Authentication failed: Invalid API key"
                )
            else:
                raise QdrantCloudError(f"Qdrant Cloud operation failed: {e}")
        except ResponseHandlingException as e:
            raise QdrantCloudConnectionError(f"Connection failed: {e}")
        except Exception as e:
            raise QdrantCloudError(f"Unexpected error deleting document chunks: {e}")

    def search_points(
        self, collection_name: str, vector: List[float], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar points in a collection

        Args:
            collection_name: Name of the collection to search
            vector: Query vector to find similar points
            limit: Maximum number of results to return

        Returns:
            List of search results with id, score, payload, and vector

        Raises:
            QdrantCloudError: If search operation fails
        """
        try:
            self.logger.debug(
                "Searching points",
                collection_name=collection_name,
                vector_dim=len(vector),
                limit=limit,
            )

            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )

            # Convert results to standard format
            results = []
            for scored_point in search_results:
                result = {
                    "id": scored_point.id,
                    "score": scored_point.score,
                    "payload": scored_point.payload or {},
                    "vector": scored_point.vector,
                }
                results.append(result)

            self.logger.debug(
                "Search completed",
                collection_name=collection_name,
                results_count=len(results),
            )

            return results

        except UnexpectedResponse as e:
            if e.status_code == 401:
                raise QdrantCloudAuthenticationError(
                    "Authentication failed: Invalid API key"
                )
            else:
                raise QdrantCloudError(f"Qdrant Cloud operation failed: {e}")
        except ResponseHandlingException as e:
            raise QdrantCloudConnectionError(f"Connection failed: {e}")
        except Exception as e:
            raise QdrantCloudError(f"Unexpected error during search: {e}")

    def health_check(self) -> bool:
        """
        Check if Qdrant Cloud service is accessible and healthy

        Returns:
            True if the service is healthy, False otherwise
        """
        try:
            # Try to get collections list as a health check
            collections = self.client.get_collections()

            self.logger.info(
                "Health check completed",
                healthy=True,
                collections_count=len(collections.collections),
            )
            return True

        except UnexpectedResponse as e:
            if e.status_code == 401:
                raise QdrantCloudAuthenticationError(
                    "Authentication failed: Invalid API key"
                )
            else:
                self.logger.warning("Health check failed", error=str(e))
                return False
        except ResponseHandlingException as e:
            raise QdrantCloudConnectionError(f"Connection failed: {e}")
        except Exception as e:
            self.logger.warning("Health check failed", error=str(e))
            return False

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary containing collection information

        Raises:
            QdrantCloudError: If operation fails
        """
        try:
            collection_info = self.client.get_collection(collection_name)

            return {
                "name": collection_name,
                "vector_size": collection_info.config.params.vectors.size,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
            }

        except UnexpectedResponse as e:
            if e.status_code == 401:
                raise QdrantCloudAuthenticationError(
                    "Authentication failed: Invalid API key"
                )
            elif e.status_code == 404:
                raise QdrantCloudError(f"Collection '{collection_name}' not found")
            else:
                raise QdrantCloudError(f"Qdrant Cloud operation failed: {e}")
        except ResponseHandlingException as e:
            raise QdrantCloudConnectionError(f"Connection failed: {e}")
        except Exception as e:
            raise QdrantCloudError(f"Unexpected error getting collection info: {e}")
