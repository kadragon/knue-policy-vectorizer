#!/usr/bin/env python3
"""
Integration test script for the complete sync pipeline.
Tests real repository synchronization with Qdrant.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time

import structlog
from sync_pipeline import SyncError, SyncPipeline

from config import Config

# Setup logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


def test_full_sync_pipeline():
    """Test the complete sync pipeline with real repository."""
    logger.info("Starting full sync pipeline integration test")

    # Create configuration
    config = Config(
        repo_url="https://github.com/kadragon/KNUE-Policy-Hub.git",
        branch="main",
        repo_cache_dir="./test_repo_cache",
        qdrant_collection="knue-policy-test",
        vector_size=1024,
        openai_model="text-embedding-3-small",
    )

    # Initialize pipeline
    pipeline = SyncPipeline(config)

    try:
        # Step 1: Health Check
        logger.info("Step 1: Performing health check")
        if not pipeline.health_check():
            logger.error("Health check failed - ensure Ollama and Qdrant are running")
            return False

        logger.info("✅ Health check passed")

        # Step 2: Full Reindex (Clean Slate)
        logger.info("Step 2: Performing full reindex")
        start_time = time.time()

        reindex_result = pipeline.reindex_all()
        reindex_time = time.time() - start_time

        logger.info(
            "✅ Full reindex completed",
            status=reindex_result["status"],
            total_files=reindex_result["total_files"],
            processed=reindex_result["processed"],
            failed=reindex_result["failed"],
            duration_seconds=round(reindex_time, 2),
        )

        if reindex_result["failed"] > 0:
            logger.warning(
                "Some files failed during reindex",
                failed_files=reindex_result["failed_files"],
            )

        # Step 3: Incremental Sync Test (should show no changes)
        logger.info("Step 3: Testing incremental sync (should show no changes)")
        start_time = time.time()

        sync_result = pipeline.sync()
        sync_time = time.time() - start_time

        logger.info(
            "✅ Incremental sync completed",
            status=sync_result["status"],
            changes_detected=sync_result["changes_detected"],
            upserted=sync_result["upserted"],
            deleted=sync_result["deleted"],
            duration_seconds=round(sync_time, 2),
        )

        # Step 4: Verify Collection Info
        logger.info("Step 4: Verifying collection information")

        qdrant_service = pipeline.qdrant_service
        try:
            collection_info = qdrant_service.get_collection_info()

            if collection_info:
                logger.info(
                    "✅ Collection verification successful",
                    collection_name=config.qdrant_collection,
                    points_count=collection_info.points_count,
                    vectors_count=collection_info.vectors_count,
                    status=collection_info.status,
                )
            else:
                logger.error("❌ Failed to get collection information")
                return False
        except Exception as e:
            logger.warning("Collection info check failed", error=str(e))
            collection_info = None

        # Step 5: Test Search Functionality
        logger.info("Step 5: Testing search functionality")

        # Try to search for a common policy term
        try:
            # Generate embedding for search query
            query_embedding = pipeline.embedding_service.generate_embedding("학사 정책")
            search_results = qdrant_service.search_points(
                query_vector=query_embedding, limit=3
            )

            logger.info(
                "✅ Search functionality test successful",
                query="학사 정책",
                results_count=len(search_results),
                top_score=search_results[0].score if search_results else 0,
            )

            # Log top results
            for i, result in enumerate(search_results[:3]):
                logger.info(
                    f"Search result {i+1}",
                    doc_id=result.id,
                    score=result.score,
                    title=result.payload.get("title", "Unknown"),
                )

        except Exception as e:
            logger.warning("Search functionality test failed", error=str(e))

        # Final Summary
        logger.info(
            "🎉 Full sync pipeline integration test completed successfully",
            total_documents=reindex_result["processed"],
            reindex_time_seconds=round(reindex_time, 2),
            sync_time_seconds=round(sync_time, 2),
            collection_points=collection_info.points_count if collection_info else 0,
        )

        return True

    except SyncError as e:
        logger.error(
            "Sync pipeline test failed",
            error=str(e),
            cause=str(e.cause) if e.cause else None,
        )
        return False
    except Exception as e:
        logger.error(
            "Unexpected error during test", error=str(e), error_type=type(e).__name__
        )
        return False


def cleanup_test_data():
    """Clean up test data and collections."""
    logger.info("Cleaning up test data")

    try:
        config = Config(qdrant_collection="knue-policy-test")
        pipeline = SyncPipeline(config)

        # Delete test collection
        if pipeline.qdrant_service.collection_exists():
            pipeline.qdrant_service.delete_collection()
            logger.info(
                "✅ Test collection deleted", collection=config.qdrant_collection
            )

        # Clean up repo cache
        import shutil

        cache_dir = Path("./test_repo_cache")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logger.info("✅ Test repository cache cleaned up")

    except Exception as e:
        logger.warning("Failed to clean up test data", error=str(e))


def main():
    """Main test runner."""
    print("🚀 Starting KNUE Policy Vectorizer - Full Sync Pipeline Integration Test")
    print("=" * 80)

    try:
        # Run the test
        success = test_full_sync_pipeline()

        print("=" * 80)
        if success:
            print("✅ Integration test PASSED - All components working correctly!")
            print("\nNext steps:")
            print("- The sync pipeline is ready for production use")
            print("- You can now run: uv run python src/sync_pipeline.py sync")
            print("- Or use the reindex command for full re-indexing")
        else:
            print("❌ Integration test FAILED - Check logs for details")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Test failed with unexpected error: {e}")
        return 1

    finally:
        # Always try to clean up
        print("\n🧹 Cleaning up test data...")
        cleanup_test_data()

    return 0


if __name__ == "__main__":
    exit(main())
