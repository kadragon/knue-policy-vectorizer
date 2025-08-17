#!/usr/bin/env python3
"""Test script to verify the Docker environment setup."""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import Config
from qdrant_service import QdrantService

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_qdrant_connectivity():
    """Test connectivity to Qdrant service."""
    try:
        # Create QdrantService with proper parameters
        qdrant_service = QdrantService(
            host="localhost", port=6333, collection_name="docker_test_collection"
        )

        # Test health check
        logger.info("Testing Qdrant connectivity...")
        health_info = qdrant_service.health_check()
        logger.info(f"Qdrant health check: {health_info}")

        if not health_info:
            logger.error("Qdrant health check failed")
            return False

        # Test collection operations
        test_collection = "docker_test_collection"

        # Clean up if exists
        if qdrant_service.collection_exists():
            qdrant_service.delete_collection()
            logger.info(f"Cleaned up existing collection: {test_collection}")

        # Create test collection
        qdrant_service.create_collection()
        logger.info(f"Created test collection: {test_collection}")

        # Verify collection exists
        assert qdrant_service.collection_exists()
        logger.info("Collection existence verified")

        # Clean up
        qdrant_service.delete_collection()
        logger.info("Test collection cleaned up")

        return True
    except Exception as e:
        logger.error(f"Qdrant connectivity test failed: {e}")
        return False


def test_docker_compose_setup():
    """Test the Docker Compose setup."""
    try:
        # Check if Qdrant service is running
        result = subprocess.run(
            ["docker-compose", "ps", "-q", "qdrant"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        if not result.stdout.strip():
            logger.error("Qdrant service is not running")
            return False

        logger.info("Qdrant service is running")

        # Test building the indexer image
        logger.info("Testing indexer Docker image build...")
        result = subprocess.run(
            ["docker", "build", "-t", "knue-indexer-test", "."],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        if result.returncode != 0:
            logger.error(f"Docker build failed: {result.stderr}")
            return False

        logger.info("Docker image built successfully")

        # Test running health check in container
        logger.info("Testing health check in Docker container...")
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network=knue-policy-vectorizer_knue-network",
                "--env",
                "QDRANT_URL=http://qdrant:6333",
                "knue-indexer-test",
                "uv",
                "run",
                "python",
                "-c",
                "from src.qdrant_service import QdrantService; from src.config import Config; "
                "import os; c = Config(); c.qdrant_url = os.getenv('QDRANT_URL', 'http://qdrant:6333'); "
                "q = QdrantService(c); print('Health:', q.health_check())",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        if result.returncode != 0:
            logger.error(f"Container health check failed: {result.stderr}")
            return False

        logger.info(f"Container health check output: {result.stdout.strip()}")

        return True
    except Exception as e:
        logger.error(f"Docker Compose test failed: {e}")
        return False


def main():
    """Run all Docker environment tests."""
    logger.info("=== Docker Environment Test Suite ===")

    tests = [
        ("Qdrant Connectivity", test_qdrant_connectivity),
        ("Docker Compose Setup", test_docker_compose_setup),
    ]

    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    logger.info("\n=== Test Results Summary ===")
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\n🎉 All Docker environment tests passed!")
        return 0
    else:
        logger.error("\n💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
