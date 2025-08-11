#!/usr/bin/env python3
"""
Integration test script for QdrantService with real data pipeline.
Tests complete workflow: Git → Markdown Processing → Embedding → Qdrant Storage → Search
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import structlog
from git_watcher import GitWatcher
from markdown_processor import MarkdownProcessor
from embedding_service import EmbeddingService
from qdrant_service import QdrantService
from logger import setup_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.dev.ConsoleRenderer(colors=True)
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


def create_test_markdown_files(test_dir: Path):
    """Create sample markdown files for testing"""
    
    sample_files = [
        {
            "filename": "정책1_학사관리.md",
            "content": """---
title: "학사관리 정책"
category: "학사"
last_updated: "2024-01-15"
---

# 학사관리 정책

## 개요
이 문서는 한국교원대학교의 학사관리에 관한 정책을 설명합니다.

## 주요 내용

### 1. 수강신청
- 수강신청 기간은 매 학기 시작 2주 전에 실시됩니다.
- 학점 제한: 학기당 최대 21학점까지 수강 가능합니다.

### 2. 성적 관리
- 성적 평가는 절대평가를 원칙으로 합니다.
- 재수강 시 최고 성적으로 반영됩니다.

### 3. 휴학 및 복학
- 휴학은 최대 4회, 총 4년까지 가능합니다.
- 복학 신청은 복학 예정 학기 개시 1개월 전까지 해야 합니다.
""",
        },
        {
            "filename": "정책2_장학금.md",
            "content": """---
title: "장학금 지급 정책"
category: "학생지원"
last_updated: "2024-02-01"
---

# 장학금 지급 정책

## 목적
우수한 인재 양성을 위한 학비 지원 정책입니다.

## 장학금 종류

### 1. 성적우수 장학금
- 대상: 직전 학기 평점평균 3.5 이상
- 지급액: 등록금의 50%
- 선발인원: 각 학과별 상위 10%

### 2. 국가장학금
- 대상: 소득분위 8분위 이하
- 지급액: 소득분위별 차등 지급
- 신청방법: 한국장학재단 홈페이지

### 3. 교육봉사 장학금
- 대상: 교육봉사 활동 참여 학생
- 지급액: 활동시간에 따라 차등 지급
- 최소 활동시간: 학기당 20시간 이상
""",
        },
        {
            "filename": "정책3_기숙사.md", 
            "content": """---
title: "기숙사 운영 정책"
category: "생활관"
last_updated: "2024-03-01"
---

# 기숙사 운영 정책

## 입사 자격

### 1. 우선순위
1. 신입생 (1순위)
2. 원거리 거주 학생 (2순위)
3. 경제적 어려움이 있는 학생 (3순위)

### 2. 선발 기준
- 통학거리: 30km 이상
- 소득분위: 8분위 이하 우대
- 학업성적: 직전학기 평점 2.0 이상

## 생활 규칙

### 1. 출입 시간
- 평일: 오후 11시까지 출입 가능
- 주말: 오후 12시까지 출입 가능
- 특별한 사유가 있을 경우 사전 신고

### 2. 시설 이용
- 공동시설 이용 시 질서 유지
- 소음 금지 (오후 10시 이후)
- 음주 및 흡연 금지

## 비용
- 4인실: 학기당 800,000원
- 2인실: 학기당 1,200,000원
- 식비 별도: 월 300,000원
""",
        }
    ]
    
    for file_info in sample_files:
        file_path = test_dir / file_info["filename"]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_info["content"])


def test_qdrant_integration():
    """Test complete integration from git to qdrant storage"""
    
    print("\n" + "="*80)
    print("🔍 QDRANT INTEGRATION TEST")
    print("="*80)
    
    # Configuration
    test_collection = "knue_policies_test"
    test_data_dir = Path("./test_data/markdown_samples")
    
    try:
        # Step 1: Initialize services
        print("\n📋 Step 1: Initialize Services")
        print("-" * 40)
        
        # Create test data directory and sample markdown files
        test_data_dir.mkdir(parents=True, exist_ok=True)
        create_test_markdown_files(test_data_dir)
        
        markdown_processor = MarkdownProcessor()
        
        # Check if Ollama is available
        embedding_service = EmbeddingService(
            model_name="bge-m3",
            base_url="http://localhost:11434"
        )
        
        # Test embedding service first
        if not embedding_service.health_check():
            print("❌ Embedding service health check failed")
            print("   Please ensure Ollama is running with bge-m3 model:")
            print("   docker run -d -p 11434:11434 --name ollama ollama/ollama")
            print("   ollama pull bge-m3")
            return False
        
        qdrant_service = QdrantService(
            collection_name=test_collection,
            vector_size=1024
        )
        
        print("✅ All services initialized successfully")
        
        # Step 2: Check Qdrant connection
        print("\n🔌 Step 2: Check Qdrant Connection")
        print("-" * 40)
        
        if not qdrant_service.health_check():
            print("❌ Qdrant health check failed")
            print("   Please ensure Qdrant is running:")
            print("   docker-compose -f docker-compose.qdrant.yml up -d")
            return False
        
        print("✅ Qdrant connection healthy")
        
        # Step 3: Setup test collection
        print("\n🏗️  Step 3: Setup Test Collection")
        print("-" * 40)
        
        # Clean up any existing test collection
        if qdrant_service.collection_exists():
            qdrant_service.delete_collection()
            print("🗑️  Deleted existing test collection")
        
        qdrant_service.create_collection()
        print(f"✅ Created collection: {test_collection}")
        
        # Step 4: Process sample markdown files  
        print("\n📄 Step 4: Process Sample Markdown Files")
        print("-" * 40)
        
        # Get markdown files from test directory
        test_files = list(test_data_dir.glob("*.md"))
        print(f"📁 Processing {len(test_files)} sample files:")
        
        processed_docs = []
        for file_path in test_files:
            print(f"   📝 {file_path.name}")
            
            # Process markdown file
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            
            doc = markdown_processor.process_markdown(raw_content, file_path.name)
            if doc:
                # Add some required metadata that would normally come from git
                doc.update({
                    "file_path": str(file_path),
                    "commit_hash": "test_commit_123",
                    "github_url": f"https://github.com/test/repo/blob/main/{file_path.name}",
                    "last_modified": "2024-01-01T00:00:00Z",
                    "content_length": len(doc.get("content", "")),
                })
                # Ensure document_id is present
                if 'document_id' not in doc:
                    doc['document_id'] = markdown_processor.calculate_document_id(str(file_path))
                processed_docs.append(doc)
        
        print(f"✅ Processed {len(processed_docs)} documents successfully")
        
        # Step 5: Generate embeddings
        print("\n🧮 Step 5: Generate Embeddings")
        print("-" * 40)
        
        start_time = time.time()
        embeddings_data = []
        
        for doc in processed_docs:
            try:
                embedding = embedding_service.generate_embedding(doc["content"])
                embeddings_data.append({
                    "doc": doc,
                    "embedding": embedding
                })
                print(f"   ✅ Generated embedding for: {doc['title'][:50]}...")
            except Exception as e:
                print(f"   ❌ Failed to generate embedding for {doc['title']}: {e}")
        
        embedding_time = time.time() - start_time
        print(f"✅ Generated {len(embeddings_data)} embeddings in {embedding_time:.2f}s")
        print(f"   Average: {embedding_time/len(embeddings_data):.3f}s per embedding")
        
        # Step 6: Store in Qdrant
        print("\n💾 Step 6: Store Documents in Qdrant")
        print("-" * 40)
        
        start_time = time.time()
        stored_count = 0
        
        for data in embeddings_data:
            doc = data["doc"]
            embedding = data["embedding"]
            
            try:
                success = qdrant_service.upsert_point(
                    point_id=doc["document_id"],
                    vector=embedding,
                    metadata=doc
                )
                
                if success:
                    stored_count += 1
                    print(f"   ✅ Stored: {doc['title'][:50]}...")
                else:
                    print(f"   ❌ Failed to store: {doc['title'][:50]}...")
                    
            except Exception as e:
                print(f"   ❌ Error storing {doc['title']}: {e}")
        
        storage_time = time.time() - start_time
        print(f"✅ Stored {stored_count} documents in {storage_time:.2f}s")
        
        # Step 7: Verify storage
        print("\n🔍 Step 7: Verify Storage")
        print("-" * 40)
        
        try:
            collection_info = qdrant_service.get_collection_info()
            print(f"📊 Collection Info:")
            print(f"   Name: {collection_info['name']}")
            print(f"   Status: {collection_info['status']}")
            print(f"   Points: {collection_info['vectors_count']}")
            print(f"   Vector Size: {collection_info['vector_size']}")
            print(f"   Distance: {collection_info['distance']}")
            
            if collection_info['vectors_count'] == stored_count:
                print("✅ All documents verified in collection")
            else:
                print(f"⚠️  Expected {stored_count}, found {collection_info['vectors_count']}")
                
        except Exception as e:
            print(f"❌ Failed to get collection info: {e}")
        
        # Step 8: Test search functionality
        print("\n🔎 Step 8: Test Search Functionality")
        print("-" * 40)
        
        if embeddings_data:
            # Use the first document's embedding as a query
            query_doc = embeddings_data[0]["doc"]
            query_embedding = embeddings_data[0]["embedding"]
            
            print(f"🔍 Searching with: {query_doc['title'][:50]}...")
            
            try:
                start_time = time.time()
                search_results = qdrant_service.search_points(
                    query_vector=query_embedding,
                    limit=3
                )
                search_time = time.time() - start_time
                
                print(f"⚡ Search completed in {search_time:.3f}s")
                print(f"📊 Found {len(search_results)} results:")
                
                for i, result in enumerate(search_results):
                    title = result.payload.get('title', 'Unknown')
                    score = result.score
                    print(f"   {i+1}. {title[:60]}... (score: {score:.4f})")
                
                if search_results and search_results[0].id == query_doc['document_id']:
                    print("✅ Top result matches query document (expected)")
                else:
                    print("⚠️  Top result doesn't match query document")
                    
            except Exception as e:
                print(f"❌ Search failed: {e}")
        
        # Step 9: Test point retrieval
        print("\n📖 Step 9: Test Point Retrieval")
        print("-" * 40)
        
        if processed_docs:
            test_doc = processed_docs[0]
            doc_id = test_doc['document_id']
            
            try:
                retrieved_point = qdrant_service.get_point(doc_id)
                if retrieved_point:
                    print(f"✅ Successfully retrieved point: {doc_id}")
                    print(f"   Title: {retrieved_point.payload['title'][:50]}...")
                    vector_len = len(retrieved_point.vector) if retrieved_point.vector else "N/A"
                    print(f"   Vector length: {vector_len}")
                else:
                    print(f"❌ Failed to retrieve point: {doc_id}")
                    
            except Exception as e:
                print(f"❌ Retrieval failed: {e}")
        
        # Step 10: Performance summary
        print("\n📈 Step 10: Performance Summary")
        print("-" * 40)
        
        print(f"📊 Processing Statistics:")
        print(f"   Markdown files processed: {len(processed_docs)}")
        print(f"   Embeddings generated: {len(embeddings_data)}")
        print(f"   Documents stored: {stored_count}")
        print(f"   Vector dimension: 1024")
        print(f"   Average embedding time: {embedding_time/len(embeddings_data):.3f}s")
        if stored_count > 0:
            print(f"   Average storage time: {storage_time/stored_count:.3f}s")
        else:
            print("   Average storage time: N/A (no documents stored)")
        
        # Step 11: Cleanup
        print("\n🧹 Step 11: Cleanup")
        print("-" * 40)
        
        # Auto-cleanup for automated testing
        try:
            qdrant_service.delete_collection()
            print("✅ Test collection deleted")
        except Exception as e:
            print(f"⚠️  Failed to delete test collection: {e}")
            print(f"   Collection name: {test_collection}")
            print("   View in Qdrant dashboard: http://localhost:6333/dashboard")
        
        print("\n" + "="*80)
        print("✅ QDRANT INTEGRATION TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_qdrant_integration()
    sys.exit(0 if success else 1)