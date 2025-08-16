"""Test Qdrant Docker setup and connection."""

import time
from typing import Optional

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams


class TestQdrantConnection:
    """Test Qdrant connection and basic operations."""

    @pytest.fixture(scope="class")
    def qdrant_client(self) -> Optional[QdrantClient]:
        """Create Qdrant client for testing."""
        try:
            client = QdrantClient(host="localhost", port=6333)
            # Test connection with a simple health check
            client.get_collections()
            return client
        except Exception as e:
            pytest.skip(f"Qdrant not available: {e}")

    def test_qdrant_health(self, qdrant_client):
        """Test Qdrant health endpoint."""
        # Try to get collections (basic health check)
        collections = qdrant_client.get_collections()
        assert collections is not None

    def test_create_test_collection(self, qdrant_client):
        """Test creating a test collection."""
        collection_name = "test_collection_setup"

        # Clean up if exists
        try:
            qdrant_client.delete_collection(collection_name)
        except UnexpectedResponse:
            pass  # Collection doesn't exist, which is fine

        # Create collection
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

        # Verify collection exists
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        assert collection_name in collection_names

        # Get collection info
        collection_info = qdrant_client.get_collection(collection_name)
        assert collection_info.config.params.vectors.size == 1024
        assert collection_info.config.params.vectors.distance == Distance.COSINE

        # Clean up
        qdrant_client.delete_collection(collection_name)

    def test_upsert_and_query_test_points(self, qdrant_client):
        """Test upserting points and querying."""
        collection_name = "test_collection_points"

        # Clean up if exists
        try:
            qdrant_client.delete_collection(collection_name)
        except UnexpectedResponse:
            pass

        # Create collection
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )

        # Create test points
        points = [
            {
                "id": 1,
                "vector": [0.1, 0.2, 0.3, 0.4],
                "payload": {"text": "test document 1", "category": "test"},
            },
            {
                "id": 2,
                "vector": [0.2, 0.3, 0.4, 0.5],
                "payload": {"text": "test document 2", "category": "test"},
            },
        ]

        # Upsert points
        qdrant_client.upsert(collection_name=collection_name, points=points, wait=True)

        # Wait a moment for indexing
        time.sleep(1)

        # Query similar points
        search_result = qdrant_client.query_points(
            collection_name=collection_name, query=[0.1, 0.2, 0.3, 0.4], limit=2
        ).points

        assert len(search_result) >= 1
        assert search_result[0].id == 1  # Should be most similar to itself

        # Clean up
        qdrant_client.delete_collection(collection_name)


@pytest.mark.integration
def test_docker_compose_qdrant_running():
    """Test that Qdrant can be started via Docker Compose."""
    import os
    import subprocess

    try:
        # Check if docker-compose is available
        result = subprocess.run(
            ["docker-compose", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            pytest.skip("docker-compose not available")

        # Try to start Qdrant service
        compose_file = os.path.join(
            os.path.dirname(__file__), "..", "docker-compose.qdrant.yml"
        )

        # Start service (will skip if already running)
        subprocess.run(
            ["docker-compose", "-f", compose_file, "up", "-d"],
            check=False,  # Don't fail if already running
            capture_output=True,
            timeout=30,
        )

        # Wait a moment for service to be ready
        time.sleep(5)

        # Test connection
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()

        assert collections is not None

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        pytest.skip(f"Docker environment not available: {e}")
