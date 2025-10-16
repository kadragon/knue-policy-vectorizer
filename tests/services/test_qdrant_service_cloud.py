"""
Tests for Qdrant Cloud service implementation
"""

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.services.qdrant_service_cloud import (
    QdrantCloudAuthenticationError,
    QdrantCloudConnectionError,
    QdrantCloudError,
    QdrantCloudService,
)
from src.utils.providers import VectorServiceInterface


class TestQdrantCloudService:
    """Test Qdrant Cloud service implementation"""

    def test_service_implements_interface(self) -> None:
        """Test that QdrantCloudService implements VectorServiceInterface"""
        with patch("src.services.qdrant_service_cloud.QdrantClient"):
            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="test-key"
            )
            assert isinstance(service, VectorServiceInterface)

    def test_service_initialization(self) -> None:
        """Test service initialization with different parameters"""
        with patch("src.services.qdrant_service_cloud.QdrantClient") as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            # Test with minimal parameters
            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="test-key"
            )
            assert service.url == "https://test.qdrant.tech"
            assert service.api_key == "test-key"
            assert service.timeout == 30

            # Test with custom parameters
            service = QdrantCloudService(
                url="https://custom.qdrant.tech",
                api_key="custom-key",
                timeout=60,
                prefer_grpc=True,
            )
            assert service.url == "https://custom.qdrant.tech"
            assert service.api_key == "custom-key"
            assert service.timeout == 60
            assert service.prefer_grpc is True

    def test_client_initialization_with_authentication(self) -> None:
        """Test Qdrant client initialization with API key authentication"""
        with patch("src.services.qdrant_service_cloud.QdrantClient") as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="test-key", timeout=45
            )

            mock_client.assert_called_once_with(
                url="https://test.qdrant.tech",
                api_key="test-key",
                timeout=45,
                prefer_grpc=False,
            )

    def test_create_collection_success(self) -> None:
        """Test successful collection creation"""
        with patch("src.services.qdrant_service_cloud.QdrantClient") as mock_client:
            mock_instance = Mock()
            mock_instance.create_collection.return_value = True
            mock_client.return_value = mock_instance

            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="test-key"
            )

            result = service.create_collection("test_collection", 1024)
            assert result is True

            # Verify the collection was created with correct parameters
            from qdrant_client.models import Distance, VectorParams

            mock_instance.create_collection.assert_called_once()
            call_args = mock_instance.create_collection.call_args
            assert call_args[1]["collection_name"] == "test_collection"
            assert call_args[1]["vectors_config"].size == 1024
            assert call_args[1]["vectors_config"].distance == Distance.COSINE

    def test_delete_collection_success(self) -> None:
        """Test successful collection deletion"""
        with patch("src.services.qdrant_service_cloud.QdrantClient") as mock_client:
            mock_instance = Mock()
            mock_instance.delete_collection.return_value = True
            mock_client.return_value = mock_instance

            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="test-key"
            )

            result = service.delete_collection("test_collection")
            assert result is True
            mock_instance.delete_collection.assert_called_once_with("test_collection")

    def test_collection_exists_true(self) -> None:
        """Test collection existence check - collection exists"""
        with patch("src.services.qdrant_service_cloud.QdrantClient") as mock_client:
            mock_instance = Mock()
            mock_collection_info = Mock()
            mock_instance.get_collection.return_value = mock_collection_info
            mock_client.return_value = mock_instance

            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="test-key"
            )

            result = service.collection_exists("test_collection")
            assert result is True
            mock_instance.get_collection.assert_called_once_with("test_collection")

    def test_collection_exists_false(self) -> None:
        """Test collection existence check - collection doesn't exist"""
        with patch("src.services.qdrant_service_cloud.QdrantClient") as mock_client:
            mock_instance = Mock()
            from qdrant_client.http.exceptions import UnexpectedResponse

            mock_response = UnexpectedResponse(
                status_code=404,
                content=json.dumps({"detail": "Collection not found"}).encode(),
                headers={},  # type: ignore[arg-type]
                reason_phrase="Not Found",
            )
            mock_instance.get_collection.side_effect = mock_response
            mock_client.return_value = mock_instance

            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="test-key"
            )

            result = service.collection_exists("nonexistent_collection")
            assert result is False

    def test_upsert_points_success(self) -> None:
        """Test successful point upsertion"""
        with patch("src.services.qdrant_service_cloud.QdrantClient") as mock_client:
            mock_instance = Mock()
            mock_instance.upsert.return_value = Mock(
                operation_id=123, status="completed"
            )
            mock_client.return_value = mock_instance

            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="test-key"
            )

            points = [
                {
                    "id": "doc1",
                    "vector": [0.1, 0.2, 0.3],
                    "payload": {"title": "Test Document", "content": "Test content"},
                }
            ]

            result = service.upsert_points("test_collection", points)
            assert result is True
            mock_instance.upsert.assert_called_once()

    def test_upsert_points_batch_chunking(self) -> None:
        """Test batch upsertion with automatic chunking"""
        with patch("src.services.qdrant_service_cloud.QdrantClient") as mock_client:
            mock_instance = Mock()
            mock_instance.upsert.return_value = Mock(
                operation_id=123, status="completed"
            )
            mock_client.return_value = mock_instance

            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="test-key", batch_size=2
            )

            # Create 5 points (should be split into 3 batches: 2, 2, 1)
            points = [
                {
                    "id": f"doc{i}",
                    "vector": [0.1, 0.2, 0.3],
                    "payload": {"title": f"Test Document {i}"},
                }
                for i in range(5)
            ]

            result = service.upsert_points("test_collection", points)
            assert result is True
            assert mock_instance.upsert.call_count == 3

    def test_delete_points_success(self) -> None:
        """Test successful point deletion"""
        with patch("src.services.qdrant_service_cloud.QdrantClient") as mock_client:
            mock_instance = Mock()
            mock_instance.delete.return_value = Mock(
                operation_id=123, status="completed"
            )
            mock_client.return_value = mock_instance

            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="test-key"
            )

            point_ids = ["doc1", "doc2", "doc3"]
            result = service.delete_points("test_collection", point_ids)
            assert result is True
            mock_instance.delete.assert_called_once()

    def test_search_points_success(self) -> None:
        """Test successful point search"""
        with patch("src.services.qdrant_service_cloud.QdrantClient") as mock_client:
            mock_instance = Mock()

            # Mock search results
            mock_scored_point = Mock()
            mock_scored_point.id = "doc1"
            mock_scored_point.score = 0.95
            mock_scored_point.payload = {
                "title": "Test Document",
                "content": "Test content",
            }
            mock_scored_point.vector = [0.1, 0.2, 0.3]

            mock_instance.search.return_value = [mock_scored_point]
            mock_client.return_value = mock_instance

            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="test-key"
            )

            query_vector = [0.1, 0.2, 0.3]
            results = service.search_points("test_collection", query_vector, limit=5)

            assert len(results) == 1
            assert results[0]["id"] == "doc1"
            assert results[0]["score"] == 0.95
            assert results[0]["payload"]["title"] == "Test Document"

            mock_instance.search.assert_called_once()

    def test_health_check_success(self) -> None:
        """Test successful health check"""
        with patch("src.services.qdrant_service_cloud.QdrantClient") as mock_client:
            mock_instance = Mock()
            mock_instance.get_collections.return_value = Mock(collections=[])
            mock_client.return_value = mock_instance

            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="test-key"
            )

            result = service.health_check()
            assert result is True
            mock_instance.get_collections.assert_called_once()

    def test_health_check_failure(self) -> None:
        """Test health check failure"""
        with patch("src.services.qdrant_service_cloud.QdrantClient") as mock_client:
            mock_instance = Mock()
            mock_instance.get_collections.side_effect = Exception("Connection failed")
            mock_client.return_value = mock_instance

            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="test-key"
            )

            result = service.health_check()
            assert result is False

    def test_authentication_error_handling(self) -> None:
        """Test handling of authentication errors"""
        with patch("src.services.qdrant_service_cloud.QdrantClient") as mock_client:
            mock_instance = Mock()
            from qdrant_client.http.exceptions import UnexpectedResponse

            auth_error = UnexpectedResponse(
                status_code=401,
                content=json.dumps({"detail": "Invalid API key"}).encode(),
                headers={},  # type: ignore[arg-type]
                reason_phrase="Unauthorized",
            )
            mock_instance.get_collections.side_effect = auth_error
            mock_client.return_value = mock_instance

            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="invalid-key"
            )

            with pytest.raises(
                QdrantCloudAuthenticationError, match="Authentication failed"
            ):
                service.health_check()

    def test_connection_error_handling(self) -> None:
        """Test handling of connection errors"""
        with patch("src.services.qdrant_service_cloud.QdrantClient") as mock_client:
            mock_instance = Mock()
            from qdrant_client.http.exceptions import ResponseHandlingException

            conn_error = ResponseHandlingException(Exception("Connection timeout"))
            mock_instance.get_collections.side_effect = conn_error
            mock_client.return_value = mock_instance

            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="test-key"
            )

            with pytest.raises(QdrantCloudConnectionError, match="Connection failed"):
                service.health_check()

    def test_generic_qdrant_error_handling(self) -> None:
        """Test handling of generic Qdrant errors"""
        with patch("src.services.qdrant_service_cloud.QdrantClient") as mock_client:
            mock_instance = Mock()
            from qdrant_client.http.exceptions import UnexpectedResponse

            generic_error = UnexpectedResponse(
                status_code=500,
                content=json.dumps({"detail": "Internal server error"}).encode(),
                headers={},  # type: ignore[arg-type]
                reason_phrase="Internal Server Error",
            )
            mock_instance.create_collection.side_effect = generic_error
            mock_client.return_value = mock_instance

            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="test-key"
            )

            with pytest.raises(QdrantCloudError, match="Qdrant Cloud operation failed"):
                service.create_collection("test_collection", 1024)

    def test_invalid_url_handling(self) -> None:
        """Test handling of invalid URLs"""
        with patch("src.services.qdrant_service_cloud.QdrantClient"):
            with pytest.raises(ValueError, match="Qdrant Cloud requires HTTPS"):
                QdrantCloudService(
                    url="http://test.qdrant.tech",  # Should be HTTPS for cloud
                    api_key="test-key",
                )

    def test_missing_api_key_handling(self) -> None:
        """Test handling of missing API key"""
        with pytest.raises(ValueError, match="API key is required"):
            QdrantCloudService(url="https://test.qdrant.tech", api_key="")

    def test_get_collection_info(self) -> None:
        """Test getting collection information"""
        with patch("src.services.qdrant_service_cloud.QdrantClient") as mock_client:
            mock_instance = Mock()
            mock_collection_info = Mock()
            mock_collection_info.config.params.vectors.size = 1024
            mock_collection_info.points_count = 100
            mock_collection_info.status = "green"
            mock_instance.get_collection.return_value = mock_collection_info
            mock_client.return_value = mock_instance

            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="test-key"
            )

            info = service.get_collection_info("test_collection")
            expected = {
                "name": "test_collection",
                "vector_size": 1024,
                "points_count": 100,
                "status": "green",
            }
            assert info == expected

    def test_batch_size_configuration(self) -> None:
        """Test batch size configuration"""
        with patch("qdrant_client.QdrantClient"):
            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="test-key", batch_size=50
            )
            assert service.batch_size == 50

    def test_prefer_grpc_configuration(self) -> None:
        """Test gRPC preference configuration"""
        with patch("src.services.qdrant_service_cloud.QdrantClient") as mock_client:
            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="test-key", prefer_grpc=True
            )

            mock_client.assert_called_once_with(
                url="https://test.qdrant.tech",
                api_key="test-key",
                timeout=30,
                prefer_grpc=True,
            )

    def test_point_validation(self) -> None:
        """Test point data validation"""
        with patch("src.services.qdrant_service_cloud.QdrantClient") as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="test-key"
            )

            # Test invalid point structure
            invalid_points = [{"id": "doc1"}]  # Missing vector

            with pytest.raises(QdrantCloudError, match="Invalid point structure"):
                service.upsert_points("test_collection", invalid_points)

    def test_url_validation(self) -> None:
        """Test URL validation for cloud service"""
        # Valid HTTPS URLs should work
        with patch("src.services.qdrant_service_cloud.QdrantClient"):
            service = QdrantCloudService(
                url="https://test.qdrant.tech", api_key="test-key"
            )
            assert service.url == "https://test.qdrant.tech"

        # HTTP URLs should be rejected for cloud service
        with patch("src.services.qdrant_service_cloud.QdrantClient"):
            with pytest.raises(ValueError, match="Qdrant Cloud requires HTTPS"):
                QdrantCloudService(url="http://test.qdrant.tech", api_key="test-key")
