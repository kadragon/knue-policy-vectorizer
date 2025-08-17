#!/usr/bin/env python3
"""
Integration test script for EmbeddingService with actual markdown documents
Tests the complete embedding generation pipeline with real data
"""
import logging
import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embedding_service import EmbeddingError, EmbeddingService
from markdown_processor import MarkdownProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """
    Run comprehensive integration tests for embedding service
    """
    logger.info("🔬 Starting Embedding Service Integration Tests")

    # Initialize services
    embedding_service = EmbeddingService()
    markdown_processor = MarkdownProcessor()

    # Test 1: Health check
    logger.info("\n📡 Test 1: Health Check")
    logger.info("Checking Ollama service connectivity...")

    if not embedding_service.health_check():
        logger.error("❌ Ollama service is not available!")
        logger.info(
            "💡 Please ensure Ollama is running: docker-compose -f docker-compose.qdrant.yml up -d"
        )
        return False

    logger.info("✅ Ollama service is healthy")
    logger.info(f"🔧 Model info: {embedding_service.get_model_info()}")

    # Test 2: Korean text embedding
    logger.info("\n🇰🇷 Test 2: Korean Text Embedding")
    korean_text = """
    한국교원대학교 정책 문서
    
    이 문서는 대학의 운영 방침과 정책을 담고 있습니다.
    교육과정, 학사관리, 연구지원 등에 대한 내용이 포함되어 있으며,
    모든 구성원이 준수해야 할 기본 원칙을 제시합니다.
    """

    try:
        start_time = time.time()
        embedding = embedding_service.generate_embedding(korean_text.strip())
        generation_time = time.time() - start_time

        logger.info(f"✅ Korean text embedding generated successfully")
        logger.info(f"📏 Embedding dimension: {len(embedding)}")
        logger.info(f"⏱️  Generation time: {generation_time:.2f}s")
        logger.info(f"🔢 Sample values: {embedding[:5]}")

        # Verify embedding dimension
        if len(embedding) != 1024:
            logger.error(f"❌ Expected 1024 dimensions, got {len(embedding)}")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to generate Korean text embedding: {e}")
        return False

    # Test 3: Token limit validation
    logger.info("\n📏 Test 3: Token Limit Validation")

    # Test normal text (under limit)
    normal_text = "한국교원대학교 " * 100  # ~200 tokens
    try:
        embedding_service.validate_token_limit(normal_text)
        logger.info("✅ Normal text passed token validation")
    except ValueError as e:
        logger.error(f"❌ Normal text failed token validation: {e}")
        return False

    # Test oversized text (over limit)
    oversized_text = "한국교원대학교 정책문서입니다. " * 2000  # ~8000+ tokens
    try:
        embedding_service.validate_token_limit(oversized_text)
        logger.error("❌ Oversized text should have failed validation")
        return False
    except ValueError:
        logger.info("✅ Oversized text correctly failed token validation")
    except Exception as e:
        logger.error(f"❌ Unexpected error in token validation: {e}")
        return False

    # Test 4: Batch embedding with multiple Korean texts
    logger.info("\n📦 Test 4: Batch Embedding")
    korean_texts = [
        "첫 번째 정책 문서: 학사 관리 규정",
        "두 번째 정책 문서: 연구윤리 지침",
        "세 번째 정책 문서: 교육과정 운영 방침",
        "네 번째 정책 문서: 학생 생활 규칙",
    ]

    try:
        start_time = time.time()
        embeddings = embedding_service.generate_embeddings_batch(korean_texts)
        batch_time = time.time() - start_time

        logger.info(f"✅ Batch embeddings generated successfully")
        logger.info(f"📦 Batch size: {len(embeddings)}")
        logger.info(f"⏱️  Batch generation time: {batch_time:.2f}s")
        logger.info(f"📊 Average time per embedding: {batch_time/len(embeddings):.2f}s")

        # Verify all embeddings have correct dimensions
        for i, emb in enumerate(embeddings):
            if len(emb) != 1024:
                logger.error(f"❌ Embedding {i} has wrong dimension: {len(emb)}")
                return False

        # Verify embeddings are different (not all the same)
        if all(embeddings[0] == emb for emb in embeddings[1:]):
            logger.error("❌ All embeddings are identical - this suggests an error")
            return False

        logger.info("✅ All batch embeddings have correct dimensions and are unique")

    except Exception as e:
        logger.error(f"❌ Failed to generate batch embeddings: {e}")
        return False

    # Test 5: Real markdown document processing and embedding
    logger.info("\n📄 Test 5: Real Markdown Document Processing")

    # Create a sample markdown document similar to KNUE policies
    sample_markdown = """---
title: "한국교원대학교 학사관리 규정"
date: "2024-01-01"
category: "학사"
---

# 한국교원대학교 학사관리 규정

## 제1장 총칙

### 제1조 (목적)
이 규정은 한국교원대학교(이하 "본교"라 한다)의 학사관리에 관한 세부사항을 정함을 목적으로 한다.

### 제2조 (적용범위)
이 규정은 본교 학부 과정의 학사관리에 적용한다.

## 제2장 수업과 학점

### 제3조 (수업일수)
1. 매 학기 수업일수는 15주 이상으로 한다.
2. 수업일수에는 시험기간을 포함하지 아니한다.

### 제4조 (학점계산)
1. 학점의 계산은 다음 각 호와 같다.
   - 강의: 매주 1시간씩 15주 강의를 1학점으로 한다.
   - 실험·실습: 매주 2시간씩 15주를 1학점으로 한다.

## 제3장 시험 및 성적

### 제5조 (시험)
1. 시험은 중간시험과 기말시험으로 구분한다.
2. 시험은 필기시험, 실기시험, 구술시험, 과제물 등으로 할 수 있다.
"""

    try:
        # Process the markdown document
        processed_doc = markdown_processor.process_markdown(
            sample_markdown, filename="sample_policy.md"
        )

        # Add git info to the processed document (simulate what would be added by GitWatcher)
        processed_doc.update(
            {
                "commit_hash": "abc123",
                "author": "정책관리팀",
                "date": "2024-01-01T00:00:00Z",
                "github_url": "https://github.com/knue/policies/blob/main/sample_policy.md",
            }
        )

        logger.info(f"✅ Processed markdown document")
        logger.info(f"📄 Title: {processed_doc['title']}")
        logger.info(f"📏 Content length: {len(processed_doc['content'])} chars")
        logger.info(f"🔢 Estimated tokens: {processed_doc['estimated_tokens']}")
        logger.info(
            f"✅ Document validation: {'PASSED' if processed_doc['is_valid'] else 'FAILED'}"
        )

        # Generate embedding for the processed content
        start_time = time.time()
        doc_embedding = embedding_service.generate_embedding(processed_doc["content"])
        doc_embedding_time = time.time() - start_time

        logger.info(f"✅ Document embedding generated successfully")
        logger.info(f"📏 Embedding dimension: {len(doc_embedding)}")
        logger.info(f"⏱️  Document embedding time: {doc_embedding_time:.2f}s")

        # Verify document embedding
        if len(doc_embedding) != 1024:
            logger.error(
                f"❌ Document embedding has wrong dimension: {len(doc_embedding)}"
            )
            return False

        # Check that embedding is not all zeros
        if all(val == 0 for val in doc_embedding):
            logger.error("❌ Document embedding is all zeros")
            return False

        logger.info("✅ Document embedding is valid and non-zero")

    except Exception as e:
        logger.error(f"❌ Failed to process markdown document: {e}")
        return False

    # Test 6: Performance benchmarking
    logger.info("\n⚡ Test 6: Performance Benchmarking")

    benchmark_texts = ["한국교원대학교 교육정책 " + str(i) * 50 for i in range(10)]

    try:
        # Single embedding benchmark
        single_times = []
        for text in benchmark_texts[:3]:
            start_time = time.time()
            embedding_service.generate_embedding(text)
            single_times.append(time.time() - start_time)

        avg_single_time = sum(single_times) / len(single_times)

        # Batch embedding benchmark
        start_time = time.time()
        batch_embeddings = embedding_service.generate_embeddings_batch(
            benchmark_texts[:3]
        )
        batch_time = time.time() - start_time

        logger.info(f"⚡ Performance Results:")
        logger.info(f"  Single embedding (avg): {avg_single_time:.3f}s")
        logger.info(f"  Batch embedding (3 docs): {batch_time:.3f}s")
        logger.info(f"  Batch efficiency: {(avg_single_time * 3 / batch_time):.2f}x")

        # Verify batch results
        if len(batch_embeddings) != 3:
            logger.error(f"❌ Expected 3 embeddings, got {len(batch_embeddings)}")
            return False

        logger.info("✅ Performance benchmarking completed")

    except Exception as e:
        logger.error(f"❌ Performance benchmarking failed: {e}")
        return False

    # Final summary
    logger.info("\n🎉 Integration Test Summary")
    logger.info("All tests passed successfully!")
    logger.info("\n✅ Verified functionality:")
    logger.info("  - Ollama service connectivity")
    logger.info("  - Korean text embedding generation")
    logger.info("  - Token limit validation")
    logger.info("  - Batch embedding processing")
    logger.info("  - Real markdown document processing")
    logger.info("  - Performance characteristics")
    logger.info("\n📊 Key metrics:")
    logger.info(f"  - Embedding dimension: 1024")
    logger.info(f"  - Average single embedding time: {avg_single_time:.3f}s")
    logger.info(
        f"  - Batch processing efficiency: {(avg_single_time * 3 / batch_time):.2f}x"
    )
    logger.info("\n🚀 The embedding service is ready for production use!")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
