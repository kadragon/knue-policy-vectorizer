"""
Qdrant vector database service for KNUE Policy Hub documents.

Handles vector storage operations including collection management,
point operations (upsert, delete, search), and error handling.
"""

from typing import Any, Dict, List, Optional, Union

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import (
    CollectionStatus,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScoredPoint,
    UpdateResult,
    UpdateStatus,
    VectorParams,
)

logger = structlog.get_logger(__name__)

# Configuration constants
MAX_CHUNK_DELETION_LIMIT = 1000  # Maximum chunks to consider for deletion safety


class QdrantError(Exception):
    """Custom exception for Qdrant operations"""

    pass


class QdrantService:
    """Service for managing Qdrant vector database operations"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "knue_policies",
        vector_size: int = 1024,
        distance: Distance = Distance.COSINE,
    ):
        """
        Initialize QdrantService

        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to work with
            vector_size: Dimension of vectors (should match embedding model)
            distance: Distance metric for similarity search
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self._client: Optional[Any] = None

        logger.info(
            "Initialized QdrantService",
            host=host,
            port=port,
            collection=collection_name,
            vector_size=vector_size,
        )

    @property
    def client(self) -> QdrantClient:
        """Lazy initialization of Qdrant client"""
        if self._client is None:
            try:
                self._client = QdrantClient(host=self.host, port=self.port)
                logger.debug("Created Qdrant client connection")
            except Exception as e:
                logger.error("Failed to connect to Qdrant", error=str(e))
                raise QdrantError(f"Failed to connect to Qdrant: {e}")
        return self._client

    def health_check(self) -> bool:
        """
        Check if Qdrant server is healthy and responsive

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            self.client.get_collections()
            logger.debug("Qdrant health check passed")
            return True
        except Exception as e:
            logger.warning("Qdrant health check failed", error=str(e))
            return False

    def collection_exists(self) -> bool:
        """
        Check if the collection exists

        Returns:
            True if collection exists, False otherwise
        """
        try:
            return self.client.collection_exists(self.collection_name)
        except Exception as e:
            logger.error("Failed to check collection existence", error=str(e))
            return False

    def create_collection(self) -> None:
        """
        Create collection if it doesn't exist

        Raises:
            QdrantError: If collection creation fails
        """
        try:
            if self.collection_exists():
                logger.info(
                    "Collection already exists", collection=self.collection_name
                )
                return

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, distance=self.distance
                ),
            )

            logger.info(
                "Created collection",
                collection=self.collection_name,
                vector_size=self.vector_size,
                distance=self.distance.value,
            )

        except Exception as e:
            logger.error("Failed to create collection", error=str(e))
            raise QdrantError(f"Failed to create collection: {e}")

    def delete_collection(self) -> None:
        """
        Delete collection if it exists

        Raises:
            QdrantError: If collection deletion fails
        """
        try:
            if not self.collection_exists():
                logger.info("Collection doesn't exist", collection=self.collection_name)
                return

            self.client.delete_collection(self.collection_name)
            logger.info("Deleted collection", collection=self.collection_name)

        except Exception as e:
            logger.error("Failed to delete collection", error=str(e))
            raise QdrantError(f"Failed to delete collection: {e}")

    def _validate_vector(self, vector: List[float]) -> None:
        """Validate vector dimensions"""
        if len(vector) != self.vector_size:
            raise QdrantError(
                f"Vector size mismatch: expected {self.vector_size}, got {len(vector)}"
            )

    def _validate_metadata(self, metadata: Dict[str, Any]) -> None:
        """Validate required metadata fields"""
        required_fields = [
            "document_id",
            "title",
            "file_path",
            "last_modified",
            "commit_hash",
            "github_url",
            "content_length",
            "estimated_tokens",
            "content",
            "chunk_index",
            "total_chunks",
            "section_title",
            "chunk_tokens",
            "is_chunk",
        ]

        missing_fields = [field for field in required_fields if field not in metadata]
        if missing_fields:
            raise QdrantError(f"Missing required metadata fields: {missing_fields}")

    def upsert_point(
        self, point_id: str, vector: List[float], metadata: Dict[str, Any]
    ) -> bool:
        """
        Insert or update a point in the collection

        Args:
            point_id: Unique identifier for the point
            vector: Embedding vector
            metadata: Associated metadata

        Returns:
            True if operation successful

        Raises:
            QdrantError: If upsert fails or validation fails
        """
        try:
            self._validate_vector(vector)
            self._validate_metadata(metadata)

            point = PointStruct(id=point_id, vector=vector, payload=metadata)

            result = self.client.upsert(
                collection_name=self.collection_name, points=[point]
            )

            if result.status == UpdateStatus.COMPLETED:
                logger.debug("Upserted point", point_id=point_id)
                return True
            else:
                logger.warning("Upsert operation pending", point_id=point_id)
                return True  # Still consider it successful

        except QdrantError:
            raise  # Re-raise validation errors
        except Exception as e:
            logger.error("Failed to upsert point", point_id=point_id, error=str(e))
            raise QdrantError(f"Failed to upsert point {point_id}: {e}")

    def upsert_points_batch(self, points_data: List[Dict[str, Any]]) -> bool:
        """
        Insert or update multiple points in batch

        Args:
            points_data: List of dictionaries with point_id, vector, metadata

        Returns:
            True if operation successful

        Raises:
            QdrantError: If batch upsert fails
        """
        try:
            points = []
            for point_data in points_data:
                point_id = point_data["point_id"]
                vector = point_data["vector"]
                metadata = point_data["metadata"]

                self._validate_vector(vector)
                self._validate_metadata(metadata)

                points.append(PointStruct(id=point_id, vector=vector, payload=metadata))

            result = self.client.upsert(
                collection_name=self.collection_name, points=points
            )

            logger.info("Batch upserted points", count=len(points))
            return result.status == UpdateStatus.COMPLETED

        except QdrantError:
            raise  # Re-raise validation errors
        except Exception as e:
            logger.error("Failed to batch upsert points", error=str(e))
            raise QdrantError(f"Failed to batch upsert points: {e}")

    def delete_point(self, point_id: str) -> bool:
        """
        Delete a point from the collection

        Args:
            point_id: ID of the point to delete

        Returns:
            True if operation successful

        Raises:
            QdrantError: If deletion fails
        """
        try:
            result = self.client.delete(
                collection_name=self.collection_name, points_selector=[point_id]
            )

            logger.debug("Deleted point", point_id=point_id)
            return result.status == UpdateStatus.COMPLETED

        except Exception as e:
            logger.error("Failed to delete point", point_id=point_id, error=str(e))
            raise QdrantError(f"Failed to delete point {point_id}: {e}")

    def delete_points_batch(self, point_ids: List[str]) -> bool:
        """
        Delete multiple points in batch

        Args:
            point_ids: List of point IDs to delete

        Returns:
            True if operation successful

        Raises:
            QdrantError: If batch deletion fails
        """
        try:
            result = self.client.delete(
                collection_name=self.collection_name, points_selector=point_ids
            )

            logger.info("Batch deleted points", count=len(point_ids))
            return result.status == UpdateStatus.COMPLETED

        except Exception as e:
            logger.error("Failed to batch delete points", error=str(e))
            raise QdrantError(f"Failed to batch delete points: {e}")

    def search_points(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[ScoredPoint]:
        """
        Search for similar points

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of scored points

        Raises:
            QdrantError: If search fails
        """
        try:
            self._validate_vector(query_vector)

            search_params = {
                "collection_name": self.collection_name,
                "query_vector": query_vector,
                "limit": limit,
            }

            if score_threshold is not None:
                search_params["score_threshold"] = score_threshold

            results = self.client.search(**search_params)

            logger.debug(
                "Searched points",
                query_size=len(query_vector),
                limit=limit,
                results_count=len(results),
            )

            return results

        except QdrantError:
            raise  # Re-raise validation errors
        except Exception as e:
            logger.error("Failed to search points", error=str(e))
            raise QdrantError(f"Failed to search points: {e}")

    def get_point(self, point_id: str) -> Optional[ScoredPoint]:
        """
        Retrieve a specific point by ID

        Args:
            point_id: ID of the point to retrieve

        Returns:
            Point if found, None otherwise

        Raises:
            QdrantError: If retrieval fails
        """
        try:
            results = self.client.retrieve(
                collection_name=self.collection_name, ids=[point_id]
            )

            if results:
                logger.debug("Retrieved point", point_id=point_id)
                return results[0]
            else:
                logger.debug("Point not found", point_id=point_id)
                return None

        except Exception as e:
            logger.error("Failed to retrieve point", point_id=point_id, error=str(e))
            raise QdrantError(f"Failed to retrieve point {point_id}: {e}")

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get collection information and statistics

        Returns:
            Dictionary with collection info

        Raises:
            QdrantError: If operation fails
        """
        try:
            info = self.client.get_collection(self.collection_name)

            return {
                "name": self.collection_name,
                "status": (
                    info.status.value
                    if hasattr(info.status, "value")
                    else str(info.status)
                ),
                "vectors_count": info.points_count,
                "vector_size": info.config.params.vectors.size,
                "distance": (
                    info.config.params.vectors.distance.value
                    if hasattr(info.config.params.vectors.distance, "value")
                    else str(info.config.params.vectors.distance)
                ),
            }

        except Exception as e:
            logger.error("Failed to get collection info", error=str(e))
            raise QdrantError(f"Failed to get collection info: {e}")

    def _find_document_chunks(self, base_document_id: str) -> List[str]:
        """
        Find all chunk IDs for a document using Qdrant scroll API

        Args:
            base_document_id: Base document ID to search for

        Returns:
            List of point IDs that belong to this document

        Raises:
            QdrantError: If search fails
        """
        try:
            point_ids = []

            # Use scroll to iterate through all points with matching document_id
            scroll_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id", match=MatchValue(value=base_document_id)
                    )
                ]
            )

            offset = None
            limit = 100  # Process in batches of 100

            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=scroll_filter,
                    limit=limit,
                    offset=offset,
                    with_payload=False,  # We only need IDs
                    with_vectors=False,  # We only need IDs
                )

                if not scroll_result[0]:  # No more points
                    break

                # Extract point IDs
                batch_ids = [point.id for point in scroll_result[0]]
                point_ids.extend(batch_ids)

                # Check safety limit
                if len(point_ids) > MAX_CHUNK_DELETION_LIMIT:
                    logger.warning(
                        "Document has excessive chunks, limiting deletion",
                        base_document_id=base_document_id,
                        found_chunks=len(point_ids),
                        limit=MAX_CHUNK_DELETION_LIMIT,
                    )
                    point_ids = point_ids[:MAX_CHUNK_DELETION_LIMIT]
                    break

                # Update offset for next iteration
                offset = scroll_result[1]

                # If offset is None, we're done (no more results)
                if offset is None:
                    break

            logger.debug(
                "Found document chunks",
                base_document_id=base_document_id,
                chunk_count=len(point_ids),
            )

            return point_ids

        except Exception as e:
            logger.error(
                "Failed to find document chunks",
                base_document_id=base_document_id,
                error=str(e),
            )
            raise QdrantError(
                f"Failed to find document chunks for {base_document_id}: {e}"
            )

    def delete_document_chunks(self, base_document_id: str) -> bool:
        """
        Delete all chunks for a document (including the base document and all chunks)

        This method now uses Qdrant's scroll API to find all chunks belonging to a document
        rather than assuming a fixed maximum number of chunks.

        Args:
            base_document_id: Base document ID (without chunk suffix)

        Returns:
            True if operation successful

        Raises:
            QdrantError: If deletion fails
        """
        try:
            # Find all point IDs that belong to this document
            point_ids_to_delete = self._find_document_chunks(base_document_id)

            if not point_ids_to_delete:
                logger.info(
                    "No chunks found for document", base_document_id=base_document_id
                )
                return True

            # Delete all found IDs in batch
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids_to_delete,
            )

            logger.info(
                "Deleted document and chunks",
                base_document_id=base_document_id,
                chunks_deleted=len(point_ids_to_delete),
            )
            return result.status == UpdateStatus.COMPLETED

        except Exception as e:
            logger.error(
                "Failed to delete document chunks",
                base_document_id=base_document_id,
                error=str(e),
            )
            raise QdrantError(
                f"Failed to delete document chunks for {base_document_id}: {e}"
            )
