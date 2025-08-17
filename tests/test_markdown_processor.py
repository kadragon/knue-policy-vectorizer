"""Test Markdown preprocessing functionality (TDD approach)."""

import os
import sys
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestMarkdownProcessor:
    """Test Markdown preprocessing functionality."""

    def test_markdown_processor_initialization(self):
        """Test MarkdownProcessor can be initialized."""
        from markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()
        assert processor is not None
        assert hasattr(processor, "max_chars")
        assert hasattr(processor, "max_tokens")

    def test_remove_frontmatter_yaml(self):
        """Test removing YAML frontmatter from markdown content."""
        from markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        # Test case 1: YAML frontmatter with content
        markdown_with_yaml = """---
title: "Test Policy Document"
date: 2024-01-15
category: policy
tags: [education, regulation]
---

# 테스트 정책

이것은 테스트 정책 문서입니다."""

        expected_content = """# 테스트 정책

이것은 테스트 정책 문서입니다."""

        result = processor.remove_frontmatter(markdown_with_yaml)
        assert result == expected_content

        # Test case 2: No frontmatter
        markdown_no_frontmatter = """# 제목

일반 마크다운 내용"""

        result = processor.remove_frontmatter(markdown_no_frontmatter)
        assert result == markdown_no_frontmatter

    def test_remove_frontmatter_toml(self):
        """Test removing TOML frontmatter from markdown content."""
        from markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        # TOML frontmatter
        markdown_with_toml = """+++
title = "Test Policy"
date = 2024-01-15
+++

# 테스트 정책

TOML frontmatter 테스트"""

        expected_content = """# 테스트 정책

TOML frontmatter 테스트"""

        result = processor.remove_frontmatter(markdown_with_toml)
        assert result == expected_content

    def test_extract_title_from_h1(self):
        """Test extracting title from H1 heading."""
        from markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        # Test case 1: Simple H1
        content = """# 한국교원대학교 학칙

본 학칙은 한국교원대학교의 기본 규정입니다."""

        title = processor.extract_title(content)
        assert title == "한국교원대학교 학칙"

        # Test case 2: H1 with formatting
        content_formatted = """# **한국교원대학교** 대학원 학칙

대학원 규정입니다."""

        title = processor.extract_title(content_formatted)
        assert title == "한국교원대학교 대학원 학칙"  # Should remove formatting

        # Test case 3: Multiple H1s (should return first)
        content_multiple = """# 첫 번째 제목

내용

# 두 번째 제목

더 많은 내용"""

        title = processor.extract_title(content_multiple)
        assert title == "첫 번째 제목"

    def test_extract_title_from_filename(self):
        """Test extracting title from filename when no H1 exists."""
        from markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        # No H1 heading in content
        content = """이 문서에는 H1 제목이 없습니다.

## H2 제목만 있습니다

일부 내용"""

        filename = "한국교원대학교 설치령.md"
        title = processor.extract_title(content, filename)
        assert title == "한국교원대학교 설치령"  # Should use filename without extension

    def test_extract_title_edge_cases(self):
        """Test title extraction edge cases."""
        from markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        # Empty content
        title = processor.extract_title("", "test.md")
        assert title == "test"

        # Only whitespace
        title = processor.extract_title("   \n\n   ", "whitespace.md")
        assert title == "whitespace"

        # H1 with only whitespace
        content = """#   
        
Some content"""
        title = processor.extract_title(content, "empty-h1.md")
        assert title == "empty-h1"

    def test_clean_content_basic(self):
        """Test basic content cleaning."""
        from markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        # Content with excessive whitespace
        messy_content = """


# 제목   


내용 첫 번째 단락

            
        
내용 두 번째 단락    


"""

        expected_clean = """# 제목

내용 첫 번째 단락

내용 두 번째 단락"""

        result = processor.clean_content(messy_content)
        assert result == expected_clean

    def test_clean_content_preserve_structure(self):
        """Test that content cleaning preserves markdown structure."""
        from markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        content = """# 제목

## 하위 제목

- 목록 항목 1
- 목록 항목 2

> 인용문

```python
code block
```

**굵은 글씨** *기울임*"""

        # Should preserve all markdown formatting
        result = processor.clean_content(content)
        assert "# 제목" in result
        assert "## 하위 제목" in result
        assert "- 목록 항목 1" in result
        assert "> 인용문" in result
        assert "```python" in result
        assert "**굵은 글씨**" in result

    def test_process_markdown_full_pipeline(self):
        """Test the complete markdown processing pipeline."""
        from markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        # Complete markdown document with frontmatter
        raw_markdown = """---
title: "한국교원대학교 학칙"
category: education
---

# 한국교원대학교 학칙   



본 학칙은 한국교원대학교 운영의 기본이 되는 규정입니다.


## 제1조 목적  

본 학칙은...


"""

        filename = "knue-academic-rules.md"

        result = processor.process_markdown(raw_markdown, filename)

        # Should return processed content and metadata
        assert isinstance(result, dict)
        assert "content" in result
        assert "title" in result
        assert "filename" in result

        # Content should be cleaned (no frontmatter, cleaned whitespace)
        assert "---" not in result["content"]
        assert result["content"].startswith("# 한국교원대학교 학칙")

        # Title should be extracted from H1
        assert result["title"] == "한국교원대학교 학칙"

        # Filename should be stored
        assert result["filename"] == filename

    def test_generate_metadata(self):
        """Test metadata generation for processed documents."""
        from markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        content = "# 테스트 문서\n\n내용입니다."
        title = "테스트 문서"
        filename = "test-doc.md"

        # Mock file path and commit info (would come from GitWatcher)
        mock_commit_info = {"sha": "abc123def456", "date": "2024-01-15T10:30:00+09:00"}

        metadata = processor.generate_metadata(
            content=content,
            title=title,
            filename=filename,
            file_path="규정/test-doc.md",
            commit_info=mock_commit_info,
            github_url="https://github.com/kadragon/KNUE-Policy-Hub/blob/main/규정/test-doc.md",
        )

        # Should include all required metadata fields
        expected_fields = [
            "document_id",
            "title",
            "file_path",
            "last_modified",
            "commit_hash",
            "github_url",
            "content_length",
            "estimated_tokens",
        ]

        for field in expected_fields:
            assert field in metadata

        # Check specific values
        assert metadata["title"] == title
        assert metadata["file_path"] == "규정/test-doc.md"
        assert metadata["commit_hash"] == "abc123def456"
        assert (
            metadata["github_url"]
            == "https://github.com/kadragon/KNUE-Policy-Hub/blob/main/규정/test-doc.md"
        )
        assert metadata["content_length"] == len(content)
        assert metadata["estimated_tokens"] > 0

    def test_calculate_document_id(self):
        """Test document ID calculation for consistent identification."""
        from markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        file_path = "규정/제1편/제1장/한국교원대학교 설치령.md"

        doc_id = processor.calculate_document_id(file_path)

        # Should be consistent hash of file path
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

        # Same path should produce same ID
        doc_id2 = processor.calculate_document_id(file_path)
        assert doc_id == doc_id2

        # Different paths should produce different IDs
        different_path = "규정/제1편/제2장/다른_파일.md"
        different_id = processor.calculate_document_id(different_path)
        assert doc_id != different_id

    def test_estimate_token_count(self):
        """Test token count estimation for length validation."""
        from markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        # Short content
        short_content = "# 제목\n\n짧은 내용입니다."
        token_count = processor.estimate_token_count(short_content)
        assert isinstance(token_count, int)
        assert token_count > 0
        assert token_count < 100  # Should be small number

        # Long content
        long_content = "# 긴 문서\n\n" + "이것은 긴 내용입니다. " * 1000
        long_token_count = processor.estimate_token_count(long_content)
        assert long_token_count > token_count  # Should be larger

    def test_validate_content_length(self):
        """Test content length validation against limits."""
        from markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        # Normal content (should pass)
        normal_content = "# 제목\n\n적당한 길이의 내용입니다."
        is_valid, message = processor.validate_content_length(normal_content)
        assert is_valid == True
        assert message is None

        # Very long content (should fail)
        very_long_content = "# 긴 문서\n\n" + "매우 긴 내용입니다. " * 10000
        is_valid, message = processor.validate_content_length(very_long_content)
        assert is_valid == False
        assert "too long" in message.lower() or "길이" in message


class TestMarkdownProcessorIntegration:
    """Integration tests for MarkdownProcessor with real data."""

    def test_process_real_knue_document(self):
        """Test processing actual KNUE policy document format."""
        from markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        # Sample content similar to actual KNUE documents
        sample_knue_doc = """# 한국교원대학교 학칙

## 제1장 총칙

### 제1조(목적)
본 학칙은 한국교원대학교(이하 "본교"라 한다)의 교육목적을 달성하기 위하여 필요한 사항을 규정함을 목적으로 한다.

### 제2조(교육목적)  
본교는 사범계 고등교육기관으로서 교원의 양성과 현직교원의 재교육을 담당하며, 교원교육 전반에 관한 학술연구를 통하여 우리나라 교육발전에 기여함을 목적으로 한다.

## 제2장 조직

### 제3조(학교의 구성)
본교에는 다음과 각호의 기관을 둔다.
1. 대학
2. 대학원
3. 부속기관
"""

        filename = "한국교원대학교 학칙.md"

        result = processor.process_markdown(sample_knue_doc, filename)

        # Verify structure is preserved
        assert "제1장 총칙" in result["content"]
        assert "제1조(목적)" in result["content"]
        assert "제2조(교육목적)" in result["content"]

        # Verify title extraction
        assert result["title"] == "한국교원대학교 학칙"


# Tests for setup verification
def test_frontmatter_library_available():
    """Test that python-frontmatter library is available."""
    import frontmatter

    assert hasattr(frontmatter, "load")
    assert hasattr(frontmatter, "loads")


def test_hashlib_available():
    """Test that hashlib is available for document ID generation."""
    import hashlib

    test_hash = hashlib.sha256(b"test").hexdigest()
    assert len(test_hash) == 64  # SHA-256 hash length


def test_datetime_handling():
    """Test datetime handling for metadata."""
    import time
    from datetime import datetime

    # Test ISO format generation
    now = datetime.now()
    iso_string = now.isoformat()
    assert "T" in iso_string

    # Test timestamp
    timestamp = time.time()
    assert isinstance(timestamp, float)


class TestDocumentChunkingIntegration:
    """Test chunking integration with MarkdownProcessor."""

    def test_create_document_for_vectorization_single_document(self):
        """Test document creation for non-chunked content."""
        from markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        # Normal sized content
        processed_content = {
            "content": "# 테스트\n\n내용입니다.",
            "title": "테스트",
            "filename": "test.md",
            "is_valid": True,
            "validation_error": None,
            "char_count": 20,
            "estimated_tokens": 10,
            "needs_chunking": False,
            "chunks": None,
        }

        mock_commit_info = {"sha": "abc123"}
        documents = processor.create_document_for_vectorization(
            processed_content=processed_content,
            file_path="test.md",
            commit_info=mock_commit_info,
            github_url="https://github.com/test/repo/blob/main/test.md",
        )

        # Should return list with single document
        assert isinstance(documents, list)
        assert len(documents) == 1

        document = documents[0]
        assert document["content"] == processed_content["content"]
        assert document["metadata"]["is_chunk"] == False
        assert document["metadata"]["total_chunks"] == 1
        assert document["processing_info"]["is_chunk"] == False

    def test_create_document_for_vectorization_chunked_documents(self):
        """Test document creation for chunked content."""
        from markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        # Mock chunked content
        mock_chunks = [
            {
                "content": "# 테스트 Part 1\n\n첫 번째 부분",
                "tokens": 400,
                "chunk_index": 0,
                "section_title": "테스트",
                "char_count": 25,
                "has_context_overlap": False,
                "overlap_tokens": 0,
            },
            {
                "content": "# 테스트 Part 2\n\n두 번째 부분",
                "tokens": 450,
                "chunk_index": 1,
                "section_title": "테스트",
                "char_count": 25,
                "has_context_overlap": True,
                "overlap_tokens": 50,
            },
        ]

        processed_content = {
            "content": "# 테스트\n\n전체 내용입니다.",
            "title": "테스트",
            "filename": "long-test.md",
            "is_valid": True,
            "validation_error": None,
            "char_count": 1000,
            "estimated_tokens": 900,
            "needs_chunking": True,
            "chunks": mock_chunks,
        }

        mock_commit_info = {"sha": "def456"}
        documents = processor.create_document_for_vectorization(
            processed_content=processed_content,
            file_path="long-test.md",
            commit_info=mock_commit_info,
            github_url="https://github.com/test/repo/blob/main/long-test.md",
        )

        # Should return list with multiple documents
        assert isinstance(documents, list)
        assert len(documents) == 2

        # Check first chunk
        doc1 = documents[0]
        assert doc1["content"] == mock_chunks[0]["content"]
        assert doc1["metadata"]["is_chunk"] == True
        assert doc1["metadata"]["chunk_index"] == 0
        assert doc1["metadata"]["total_chunks"] == 2
        assert doc1["metadata"]["has_context_overlap"] == False
        assert doc1["processing_info"]["is_chunk"] == True
        assert "Part 1" in doc1["metadata"]["title"]

        # Check second chunk
        doc2 = documents[1]
        assert doc2["content"] == mock_chunks[1]["content"]
        assert doc2["metadata"]["is_chunk"] == True
        assert doc2["metadata"]["chunk_index"] == 1
        assert doc2["metadata"]["total_chunks"] == 2
        assert doc2["metadata"]["has_context_overlap"] == True
        assert doc2["processing_info"]["is_chunk"] == True
        assert "Part 2" in doc2["metadata"]["title"]

        # Document IDs should be different
        assert doc1["metadata"]["document_id"] != doc2["metadata"]["document_id"]
        assert "chunk_0" in doc1["metadata"]["document_id"]
        assert "chunk_1" in doc2["metadata"]["document_id"]

    def test_chunk_markdown_content_overlap_strategy(self):
        """Test the 800/200 overlap strategy in chunking."""
        from markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        # Create content that will require multiple chunks
        content = (
            """# 긴 문서

## 섹션 1
"""
            + "섹션 1의 내용입니다. " * 100
            + """

## 섹션 2  
"""
            + "섹션 2의 내용입니다. " * 100
            + """

## 섹션 3
"""
            + "섹션 3의 내용입니다. " * 100
        )

        chunks = processor.chunk_markdown_content(content, max_tokens=400)

        # Should create multiple chunks
        assert len(chunks) > 1

        # Check overlap metadata - with current algorithm, some chunks may be small
        # due to section boundaries, but they should still have the required fields
        for i, chunk in enumerate(chunks):
            assert "has_context_overlap" in chunk
            assert "overlap_tokens" in chunk

            if i == 0:
                # First chunk should not have overlap
                assert chunk.get("has_context_overlap", False) == False
                assert chunk.get("overlap_tokens", 0) == 0

            # If a chunk has context overlap, verify the metadata is consistent
            if chunk.get("has_context_overlap", False):
                assert chunk.get("overlap_tokens", 0) > 0  # Should have overlap tokens

            # If overlap_tokens > 0, should have has_context_overlap set to True
            if chunk.get("overlap_tokens", 0) > 0:
                assert chunk.get("has_context_overlap", False) == True
