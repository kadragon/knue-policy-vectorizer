#!/usr/bin/env python3
"""Script to verify Qdrant setup and connection."""

import sys
import time

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams


def main():
    """Main verification function."""
    print("🔍 Verifying Qdrant setup...")

    try:
        # Connect to Qdrant
        print("📡 Connecting to Qdrant...")
        client = QdrantClient(host="localhost", port=6333)

        # Get basic info
        collections = client.get_collections()
        print(
            f"✅ Connected successfully! Found {len(collections.collections)} existing collections"
        )

        # Test collection creation
        test_collection = "verification_test"
        print(f"🧪 Testing collection creation: {test_collection}")

        # Clean up if exists
        try:
            client.delete_collection(test_collection)
            print("🗑️  Cleaned up existing test collection")
        except UnexpectedResponse:
            pass

        # Create test collection
        client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        print("✅ Test collection created successfully")

        # Verify collection
        collection_info = client.get_collection(test_collection)
        print(f"📊 Collection info:")
        print(f"   - Vector size: {collection_info.config.params.vectors.size}")
        print(f"   - Distance: {collection_info.config.params.vectors.distance}")

        # Test basic upsert
        print("📝 Testing point insertion...")
        test_points = [
            {
                "id": 1,
                "vector": [0.1] * 1024,  # Simple test vector
                "payload": {"test": True, "text": "test document"},
            }
        ]

        client.upsert(collection_name=test_collection, points=test_points, wait=True)
        print("✅ Point inserted successfully")

        # Wait for indexing
        time.sleep(1)

        # Test query
        print("🔍 Testing similarity search...")
        results = client.query_points(
            collection_name=test_collection,
            query=[0.1] * 1024,  # Query with same vector
            limit=1,
        )

        if results.points:
            print(f"✅ Query successful! Found {len(results.points)} points")
            print(f"   - Score: {results.points[0].score:.4f}")
        else:
            print("⚠️  Query returned no results")

        # Clean up
        client.delete_collection(test_collection)
        print("🗑️  Test collection cleaned up")

        print("\n🎉 All Qdrant verification tests passed!")
        print("💡 Qdrant is ready for the KNUE Policy Vectorizer!")

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        print("\n🛠️  Make sure Qdrant is running:")
        print("   docker-compose -f docker-compose.qdrant.yml up -d")
        sys.exit(1)


if __name__ == "__main__":
    main()
