"""
Test suite for QdrantService class
Tests for Qdrant vector store operations including collection management and point operations.
"""

from unittest.mock import MagicMock, AsyncMock, Mock, patch

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import (
    CollectionStatus,
    Distance,
    PointStruct,
    UpdateResult,
    UpdateStatus,
    VectorParams,
)

from src.services.qdrant_service import QdrantError, QdrantService


class TestQdrantServiceInit:
    """Test QdrantService initialization and connection"""

    def test_init_with_default_params(self) -> None:
        """Test QdrantService initialization with default parameters"""
        service = QdrantService()
        assert service.host == "localhost"
        assert service.port == 6333
        assert service.collection_name == "knue_policies"
        assert service.vector_size == 1024
        assert service.distance == Distance.COSINE

    def test_init_with_custom_params(self) -> None:
        """Test QdrantService initialization with custom parameters"""
        service = QdrantService(
            host="custom-host",
            port=9999,
            collection_name="custom_collection",
            vector_size=512,
        )
        assert service.host == "custom-host"
        assert service.port == 9999
        assert service.collection_name == "custom_collection"
        assert service.vector_size == 512

    @patch("src.services.qdrant_service.QdrantClient")
    def test_client_property_lazy_initialization(self, mock_qdrant_client: MagicMock) -> None:
        """Test that Qdrant client is initialized lazily"""
        service = QdrantService()

        # Client should not be initialized yet
        mock_qdrant_client.assert_not_called()

        # Access client property
        client = service.client

        # Now client should be initialized
        mock_qdrant_client.assert_called_once_with(host="localhost", port=6333)
        assert client == mock_qdrant_client.return_value


class TestQdrantServiceHealth:
    """Test health check functionality"""

    @patch("src.services.qdrant_service.QdrantClient")
    def test_health_check_success(self, mock_qdrant_client: MagicMock) -> None:
        """Test successful health check"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.get_collections.return_value = Mock()

        service = QdrantService()
        result = service.health_check()

        assert result is True
        mock_client.get_collections.assert_called_once()

    @patch("src.services.qdrant_service.QdrantClient")
    def test_health_check_failure(self, mock_qdrant_client: MagicMock) -> None:
        """Test health check failure"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.get_collections.side_effect = Exception("Connection failed")

        service = QdrantService()
        result = service.health_check()

        assert result is False


class TestQdrantServiceCollectionManagement:
    """Test collection management operations"""

    @patch("src.services.qdrant_service.QdrantClient")
    def test_collection_exists_true(self, mock_qdrant_client: MagicMock) -> None:
        """Test collection_exists returns True when collection exists"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = True

        service = QdrantService()
        result = service.collection_exists()

        assert result is True
        mock_client.collection_exists.assert_called_once_with("knue_policies")

    @patch("src.services.qdrant_service.QdrantClient")
    def test_collection_exists_false(self, mock_qdrant_client: MagicMock) -> None:
        """Test collection_exists returns False when collection doesn't exist"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = False

        service = QdrantService()
        result = service.collection_exists()

        assert result is False

    @patch("src.services.qdrant_service.QdrantClient")
    def test_create_collection_success(self, mock_qdrant_client: MagicMock) -> None:
        """Test successful collection creation"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = False

        service = QdrantService()
        service.create_collection()

        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args

        assert call_args[1]["collection_name"] == "knue_policies"
        assert isinstance(call_args[1]["vectors_config"], VectorParams)
        assert call_args[1]["vectors_config"].size == 1024
        assert call_args[1]["vectors_config"].distance == Distance.COSINE

    @patch("src.services.qdrant_service.QdrantClient")
    def test_create_collection_already_exists(self, mock_qdrant_client: MagicMock) -> None:
        """Test collection creation when collection already exists"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = True

        service = QdrantService()
        service.create_collection()

        # Should not call create_collection if collection exists
        mock_client.create_collection.assert_not_called()

    @patch("src.services.qdrant_service.QdrantClient")
    def test_create_collection_failure(self, mock_qdrant_client: MagicMock) -> None:
        """Test collection creation failure"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = False
        mock_client.create_collection.side_effect = Exception("Creation failed")

        service = QdrantService()

        with pytest.raises(QdrantError, match="Failed to create collection"):
            service.create_collection()

    @patch("src.services.qdrant_service.QdrantClient")
    def test_delete_collection_success(self, mock_qdrant_client: MagicMock) -> None:
        """Test successful collection deletion"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = True

        service = QdrantService()
        service.delete_collection()

        mock_client.delete_collection.assert_called_once_with("knue_policies")

    @patch("src.services.qdrant_service.QdrantClient")
    def test_delete_collection_not_exists(self, mock_qdrant_client: MagicMock) -> None:
        """Test collection deletion when collection doesn't exist"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = False

        service = QdrantService()
        service.delete_collection()

        # Should not call delete_collection if collection doesn't exist
        mock_client.delete_collection.assert_not_called()


class TestQdrantServicePointOperations:
    """Test point CRUD operations"""

    def setup_method(self) -> None:
        """Setup test data"""
        self.sample_vector = np.random.rand(1024).tolist()
        self.sample_metadata = {
            "document_id": "test_doc_1",
            "title": "Test Document",
            "file_path": "test/document.md",
            "last_modified": "2024-01-01T00:00:00Z",
            "commit_hash": "abc123",
            "github_url": "https://github.com/test/repo/blob/main/test/document.md",
            "content_length": 1500,
            "estimated_tokens": 375,
            "content": "Test document content",
            "chunk_index": 0,
            "total_chunks": 1,
            "section_title": "Test Document",
            "chunk_tokens": 375,
            "is_chunk": False,
        }

    @patch("src.services.qdrant_service.QdrantClient")
    def test_upsert_point_success(self, mock_qdrant_client: MagicMock) -> None:
        """Test successful point upsert"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.upsert.return_value = UpdateResult(
            operation_id=1, status=UpdateStatus.COMPLETED
        )

        service = QdrantService()
        result = service.upsert_point(
            point_id="test_doc_1",
            vector=self.sample_vector,
            metadata=self.sample_metadata,
        )

        assert result is True
        mock_client.upsert.assert_called_once()

        call_args = mock_client.upsert.call_args
        assert call_args[1]["collection_name"] == "knue_policies"
        assert len(call_args[1]["points"]) == 1

        point = call_args[1]["points"][0]
        assert isinstance(point, PointStruct)
        assert point.id == "test_doc_1"
        assert point.vector == self.sample_vector
        assert point.payload == self.sample_metadata

    @patch("src.services.qdrant_service.QdrantClient")
    def test_upsert_points_batch_success(self, mock_qdrant_client: MagicMock) -> None:
        """Test successful batch point upsert"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.upsert.return_value = UpdateResult(
            operation_id=1, status=UpdateStatus.COMPLETED
        )

        points_data = [
            {
                "point_id": f"test_doc_{i}",
                "vector": np.random.rand(1024).tolist(),
                "metadata": {**self.sample_metadata, "document_id": f"test_doc_{i}"},
            }
            for i in range(3)
        ]

        service = QdrantService()
        result = service.upsert_points_batch(points_data)

        assert result is True
        mock_client.upsert.assert_called_once()

        call_args = mock_client.upsert.call_args
        assert len(call_args[1]["points"]) == 3

    @patch("src.services.qdrant_service.QdrantClient")
    def test_upsert_point_invalid_vector_size(self, mock_qdrant_client: MagicMock) -> None:
        """Test upsert with invalid vector size"""
        service = QdrantService()

        invalid_vector = [1.0, 2.0, 3.0]  # Wrong size

        with pytest.raises(QdrantError, match="Vector size mismatch"):
            service.upsert_point(
                point_id="test_doc_1",
                vector=invalid_vector,
                metadata=self.sample_metadata,
            )

    @patch("src.services.qdrant_service.QdrantClient")
    def test_upsert_point_missing_metadata(self, mock_qdrant_client: MagicMock) -> None:
        """Test upsert with missing required metadata"""
        service = QdrantService()

        incomplete_metadata = {"title": "Test"}  # Missing required fields

        with pytest.raises(QdrantError, match="Missing required metadata"):
            service.upsert_point(
                point_id="test_doc_1",
                vector=self.sample_vector,
                metadata=incomplete_metadata,
            )

    @patch("src.services.qdrant_service.QdrantClient")
    def test_delete_point_success(self, mock_qdrant_client: MagicMock) -> None:
        """Test successful point deletion"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.delete.return_value = UpdateResult(
            operation_id=1, status=UpdateStatus.COMPLETED
        )

        service = QdrantService()
        result = service.delete_point("test_doc_1")

        assert result is True
        mock_client.delete.assert_called_once_with(
            collection_name="knue_policies", points_selector=["test_doc_1"]
        )

    @patch("src.services.qdrant_service.QdrantClient")
    def test_delete_points_batch_success(self, mock_qdrant_client: MagicMock) -> None:
        """Test successful batch point deletion"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.delete.return_value = UpdateResult(
            operation_id=1, status=UpdateStatus.COMPLETED
        )

        point_ids = ["test_doc_1", "test_doc_2", "test_doc_3"]

        service = QdrantService()
        result = service.delete_points_batch(point_ids)

        assert result is True
        mock_client.delete.assert_called_once_with(
            collection_name="knue_policies", points_selector=point_ids
        )


class TestQdrantServiceSearch:
    """Test search and retrieval operations"""

    @patch("src.services.qdrant_service.QdrantClient")
    def test_search_points_success(self, mock_qdrant_client: MagicMock) -> None:
        """Test successful point search"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        # Mock search result
        mock_search_result = [
            Mock(id="test_doc_1", score=0.95, payload={"title": "Test Doc 1"}),
            Mock(id="test_doc_2", score=0.88, payload={"title": "Test Doc 2"}),
        ]
        mock_client.search.return_value = mock_search_result

        query_vector = np.random.rand(1024).tolist()

        service = QdrantService()
        results = service.search_points(query_vector, limit=5)

        assert len(results) == 2
        assert results[0].id == "test_doc_1"
        assert results[0].score == 0.95

        mock_client.search.assert_called_once_with(
            collection_name="knue_policies", query_vector=query_vector, limit=5
        )

    @patch("src.services.qdrant_service.QdrantClient")
    def test_get_point_success(self, mock_qdrant_client: MagicMock) -> None:
        """Test successful point retrieval"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        mock_point = Mock(
            id="test_doc_1", vector=[1.0] * 1024, payload={"title": "Test Document"}
        )
        mock_client.retrieve.return_value = [mock_point]

        service = QdrantService()
        result = service.get_point("test_doc_1")

        assert result == mock_point
        mock_client.retrieve.assert_called_once_with(
            collection_name="knue_policies", ids=["test_doc_1"]
        )

    @patch("src.services.qdrant_service.QdrantClient")
    def test_get_point_not_found(self, mock_qdrant_client: MagicMock) -> None:
        """Test point retrieval when point doesn't exist"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.retrieve.return_value = []

        service = QdrantService()
        result = service.get_point("nonexistent_doc")

        assert result is None


class TestQdrantServiceErrorHandling:
    """Test error handling scenarios"""

    @patch("src.services.qdrant_service.QdrantClient")
    def test_client_connection_error(self, mock_qdrant_client: MagicMock) -> None:
        """Test handling of client connection errors"""
        mock_qdrant_client.side_effect = Exception("Connection refused")

        service = QdrantService()

        with pytest.raises(QdrantError, match="Failed to connect to Qdrant"):
            _ = service.client

    @patch("src.services.qdrant_service.QdrantClient")
    def test_upsert_operation_failure(self, mock_qdrant_client: MagicMock) -> None:
        """Test handling of upsert operation failures"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.upsert.side_effect = ResponseHandlingException(Exception("Server error"))

        service = QdrantService()
        vector = np.random.rand(1024).tolist()
        metadata = {
            "document_id": "test_doc_1",
            "title": "Test Document",
            "file_path": "test/document.md",
            "last_modified": "2024-01-01T00:00:00Z",
            "commit_hash": "abc123",
            "github_url": "https://github.com/test/repo/blob/main/test/document.md",
            "content_length": 1500,
            "estimated_tokens": 375,
            "content": "Test document content",
            "chunk_index": 0,
            "total_chunks": 1,
            "section_title": "Test Document",
            "chunk_tokens": 375,
            "is_chunk": False,
        }

        with pytest.raises(QdrantError, match="Failed to upsert point"):
            service.upsert_point("test_doc_1", vector, metadata)

    @patch("src.services.qdrant_service.QdrantClient")
    def test_delete_operation_failure(self, mock_qdrant_client: MagicMock) -> None:
        """Test handling of delete operation failures"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.delete.side_effect = ResponseHandlingException(Exception("Server error"))

        service = QdrantService()

        with pytest.raises(QdrantError, match="Failed to delete point"):
            service.delete_point("test_doc_1")


class TestQdrantServiceIntegration:
    """Integration-style tests with more realistic scenarios"""

    @patch("src.services.qdrant_service.QdrantClient")
    def test_full_lifecycle_workflow(self, mock_qdrant_client: MagicMock) -> None:
        """Test complete workflow: create collection, upsert points, search, delete"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        # Setup mocks for successful operations
        mock_client.collection_exists.return_value = False
        mock_client.upsert.return_value = UpdateResult(
            operation_id=1, status=UpdateStatus.COMPLETED
        )
        mock_client.search.return_value = [
            Mock(id="doc1", score=0.95, payload={"title": "Document 1"})
        ]
        mock_client.delete.return_value = UpdateResult(
            operation_id=2, status=UpdateStatus.COMPLETED
        )

        service = QdrantService()

        # Create collection
        service.create_collection()
        mock_client.create_collection.assert_called_once()

        # Upsert point
        vector = np.random.rand(1024).tolist()
        metadata = {
            "document_id": "doc1",
            "title": "Document 1",
            "file_path": "docs/doc1.md",
            "last_modified": "2024-01-01T00:00:00Z",
            "commit_hash": "abc123",
            "github_url": "https://github.com/test/repo/blob/main/docs/doc1.md",
            "content_length": 1500,
            "estimated_tokens": 375,
            "content": "Test document content",
            "chunk_index": 0,
            "total_chunks": 1,
            "section_title": "Document 1",
            "chunk_tokens": 375,
            "is_chunk": False,
        }

        result = service.upsert_point("doc1", vector, metadata)
        assert result is True

        # Search
        results = service.search_points(vector, limit=1)
        assert len(results) == 1
        assert results[0].id == "doc1"

        # Delete
        result = service.delete_point("doc1")
        assert result is True


class TestQdrantServiceDocumentDeletion:
    """Test document chunk deletion operations"""

    @patch("src.services.qdrant_service.QdrantClient")
    def test_find_document_chunks_success(self, mock_qdrant_client: MagicMock) -> None:
        """Test successful document chunk finding"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        # Mock scroll results - single batch with all chunks
        mock_chunks = [
            Mock(id="doc1"),
            Mock(id="doc1_chunk_0"),
            Mock(id="doc1_chunk_1"),
        ]
        mock_client.scroll.return_value = (mock_chunks, None)

        service = QdrantService()
        result = service._find_document_chunks("doc1")

        assert result == ["doc1", "doc1_chunk_0", "doc1_chunk_1"]
        mock_client.scroll.assert_called_once()

    @patch("src.services.qdrant_service.QdrantClient")
    def test_find_document_chunks_multiple_batches(self, mock_qdrant_client: MagicMock) -> None:
        """Test finding chunks across multiple scroll batches"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        # Mock scroll results - multiple batches
        mock_scroll_results = [
            ([Mock(id="doc1"), Mock(id="doc1_chunk_0")], "next_offset"),
            ([Mock(id="doc1_chunk_1"), Mock(id="doc1_chunk_2")], None),  # Last batch
        ]
        mock_client.scroll.side_effect = mock_scroll_results

        service = QdrantService()
        result = service._find_document_chunks("doc1")

        assert result == ["doc1", "doc1_chunk_0", "doc1_chunk_1", "doc1_chunk_2"]
        assert mock_client.scroll.call_count == 2

    @patch("src.services.qdrant_service.QdrantClient")
    def test_find_document_chunks_no_results(self, mock_qdrant_client: MagicMock) -> None:
        """Test finding chunks when none exist"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.scroll.return_value = ([], None)

        service = QdrantService()
        result = service._find_document_chunks("nonexistent_doc")

        assert result == []

    @patch("src.services.qdrant_service.QdrantClient")
    def test_find_document_chunks_limit_exceeded(self, mock_qdrant_client: MagicMock) -> None:
        """Test chunk finding with safety limit"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        # Generate more chunks than the limit
        from src.services.qdrant_service import MAX_CHUNK_DELETION_LIMIT

        chunks = [
            Mock(id=f"doc1_chunk_{i}") for i in range(MAX_CHUNK_DELETION_LIMIT + 100)
        ]
        mock_client.scroll.return_value = (chunks, None)

        service = QdrantService()
        result = service._find_document_chunks("doc1")

        assert len(result) == MAX_CHUNK_DELETION_LIMIT

    @patch("src.services.qdrant_service.QdrantClient")
    def test_delete_document_chunks_success(self, mock_qdrant_client: MagicMock) -> None:
        """Test successful document chunks deletion"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        # Mock finding chunks
        mock_client.scroll.return_value = (
            [Mock(id="doc1"), Mock(id="doc1_chunk_0"), Mock(id="doc1_chunk_1")],
            None,
        )

        # Mock deletion
        mock_client.delete.return_value = UpdateResult(
            operation_id=1, status=UpdateStatus.COMPLETED
        )

        service = QdrantService()
        result = service.delete_document_chunks("doc1")

        assert result is True
        mock_client.delete.assert_called_once_with(
            collection_name="knue_policies",
            points_selector=["doc1", "doc1_chunk_0", "doc1_chunk_1"],
        )

    @patch("src.services.qdrant_service.QdrantClient")
    def test_delete_document_chunks_no_chunks(self, mock_qdrant_client: MagicMock) -> None:
        """Test deletion when no chunks exist"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.scroll.return_value = ([], None)

        service = QdrantService()
        result = service.delete_document_chunks("nonexistent_doc")

        assert result is True
        mock_client.delete.assert_not_called()

    @patch("src.services.qdrant_service.QdrantClient")
    def test_delete_document_chunks_failure(self, mock_qdrant_client: MagicMock) -> None:
        """Test deletion failure handling"""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.scroll.side_effect = Exception("Scroll failed")

        service = QdrantService()

        with pytest.raises(QdrantError, match="Failed to delete document chunks"):
            service.delete_document_chunks("doc1")
