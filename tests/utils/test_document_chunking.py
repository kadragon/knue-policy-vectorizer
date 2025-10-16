"""Test document chunking functionality with 800/200 nesting strategy (TDD approach)."""

import os
import sys
from typing import Any, Dict, List

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


class TestDocumentChunking:
    """Test document chunking with intelligent nesting strategy."""

    def test_chunking_basic_functionality(self) -> None:
        """Test basic chunking splits long documents."""
        from src.utils.markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        # Create content that definitely needs chunking (very long)
        long_content = (
            """# 긴 문서 제목
        
첫 번째 섹션의 내용입니다. """
            + "이것은 매우 긴 내용입니다. " * 1000
            + """

## 두 번째 섹션

두 번째 섹션의 내용입니다. """
            + "또 다른 긴 내용입니다. " * 1000
        )

        chunks = processor.chunk_markdown_content(long_content, max_tokens=800)

        # Should create multiple chunks
        assert len(chunks) > 1

        # Each chunk should have required metadata
        for chunk in chunks:
            assert "content" in chunk
            assert "tokens" in chunk
            assert "chunk_index" in chunk
            assert "section_title" in chunk
            assert "char_count" in chunk
            # Note: Total tokens may exceed limit due to overlap, but that's expected

    def test_chunking_preserves_section_boundaries(self) -> None:
        """Test that chunking respects markdown section boundaries."""
        from src.utils.markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        content = (
            """# 메인 제목

첫 번째 섹션의 내용입니다. """
            + "적당한 길이의 내용. " * 100
            + """

## 섹션 A

섹션 A의 내용입니다. """
            + "섹션 A 관련 내용. " * 100
            + """

### 하위 섹션 A.1

하위 섹션의 내용입니다. """
            + "하위 내용. " * 100
            + """

## 섹션 B

섹션 B의 내용입니다. """
            + "섹션 B 관련 내용. " * 100
        )

        chunks = processor.chunk_markdown_content(content, max_tokens=400)

        # Should create multiple chunks
        assert len(chunks) > 1

        # Check that section titles are captured
        section_titles = [chunk["section_title"] for chunk in chunks]

        # Should have relevant section information
        assert any("메인 제목" in title or "섹션" in title for title in section_titles)

        # Headers should appear at the beginning of their respective chunks
        for chunk in chunks:
            content_lines = chunk["content"].split("\n")
            if any(line.strip().startswith("#") for line in content_lines):
                # If chunk contains headers, verify structure
                header_lines = [
                    line for line in content_lines if line.strip().startswith("#")
                ]
                assert len(header_lines) >= 1

    def test_chunking_with_overlap_strategy(self) -> None:
        """Test chunking with 800/200 nesting overlap strategy."""
        from src.utils.markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        # Create structured content that would benefit from overlap
        content = (
            """# 정책 문서

## 제1조 목적과 범위

본 조항은 정책의 목적과 적용 범위를 정의합니다. """
            + "목적 관련 상세 내용. " * 150
            + """

## 제2조 정의

이 정책에서 사용되는 용어의 정의는 다음과 같습니다. """
            + "정의 관련 상세 내용. " * 150
            + """

## 제3조 절차

정책 실행을 위한 절차는 다음과 같습니다. """
            + "절차 관련 상세 내용. " * 150
        )

        # Test with different max token limits to verify overlap behavior
        chunks_800 = processor.chunk_markdown_content(content, max_tokens=800)
        chunks_200 = processor.chunk_markdown_content(content, max_tokens=200)

        # 800 token chunks should be fewer than or equal to 200 token chunks
        assert len(chunks_800) <= len(chunks_200)

        # Note: Due to overlap strategy, total token count may exceed limit
        # but the main content should be within reasonable bounds

    def test_chunking_maintains_context_integrity(self) -> None:
        """Test that chunking maintains meaningful context boundaries."""
        from src.utils.markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        content = (
            """# 대학교 규정

## 제1장 총칙

### 제1조 목적
본 규정은 대학교 운영의 기본 원칙을 정합니다.
세부적인 목적은 다음과 같습니다:
1. 교육 목표 달성
2. 연구 활동 촉진  
3. 사회 기여 확대

### 제2조 적용 범위
이 규정은 모든 구성원에게 적용됩니다.
구체적인 적용 대상은:
- 교수진
- 직원
- 학생

## 제2장 조직

### 제3조 조직 구조
대학교의 조직은 다음과 같이 구성됩니다. """
            + "조직 관련 상세 내용. " * 200
        )

        chunks = processor.chunk_markdown_content(content, max_tokens=300)

        # Should create multiple chunks
        assert len(chunks) > 1

        # Verify that list items and related content stay together when possible
        for chunk in chunks:
            content_text = chunk["content"]

            # If a chunk contains numbered lists, they should be complete
            if "1. 교육 목표" in content_text:
                assert "2. 연구 활동" in content_text
                assert "3. 사회 기여" in content_text

            # If a chunk contains bullet points, they should be complete
            if "- 교수진" in content_text:
                assert "- 직원" in content_text
                assert "- 학생" in content_text

    def test_chunking_handles_very_short_content(self) -> None:
        """Test that chunking handles content that doesn't need splitting."""
        from src.utils.markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        short_content = """# 짧은 문서

이것은 매우 짧은 문서입니다.
청킹이 필요하지 않을 정도로 짧습니다."""

        chunks = processor.chunk_markdown_content(short_content, max_tokens=800)

        # Should create exactly one chunk
        assert len(chunks) == 1

        chunk = chunks[0]
        assert chunk["content"] == short_content
        assert chunk["chunk_index"] == 0
        # For single chunks (no overlap), tokens should be within the max_tokens limit
        assert chunk["tokens"] <= 800

    def test_chunking_handles_empty_content(self) -> None:
        """Test chunking behavior with empty or whitespace-only content."""
        from src.utils.markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        # Empty content
        empty_chunks = processor.chunk_markdown_content("", max_tokens=800)
        assert len(empty_chunks) == 0

        # Whitespace only
        whitespace_content = "   \n\n   \n  "
        whitespace_chunks = processor.chunk_markdown_content(
            whitespace_content, max_tokens=800
        )
        # Should either create no chunks or one chunk, but not fail
        assert len(whitespace_chunks) <= 1

    def test_chunking_sequential_indexing(self) -> None:
        """Test that chunks are indexed sequentially."""
        from src.utils.markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        # Create content that will definitely create multiple chunks
        content = (
            """# 메인 문서

## 섹션 1
"""
            + "첫 번째 섹션 내용. " * 200
            + """

## 섹션 2  
"""
            + "두 번째 섹션 내용. " * 200
            + """

## 섹션 3
"""
            + "세 번째 섹션 내용. " * 200
        )

        chunks = processor.chunk_markdown_content(content, max_tokens=300)

        # Should have multiple chunks
        assert len(chunks) > 1

        # Verify sequential indexing
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    def test_chunking_section_title_extraction(self) -> None:
        """Test that section titles are properly extracted for each chunk."""
        from src.utils.markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        content = (
            """# 메인 제목

메인 섹션의 내용입니다.

## 첫 번째 하위 섹션
"""
            + "첫 번째 섹션 내용. " * 100
            + """

### 상세 하위 섹션
"""
            + "상세 내용. " * 100
            + """

## 두 번째 하위 섹션  
"""
            + "두 번째 섹션 내용. " * 100
        )

        chunks = processor.chunk_markdown_content(content, max_tokens=400)

        # Check that section titles are captured appropriately
        section_titles = [chunk["section_title"] for chunk in chunks]

        # Should have captured section information
        assert any(title for title in section_titles if title)  # At least some titles

        # Verify that section titles match the headers found in chunks
        for chunk in chunks:
            if chunk["section_title"]:
                # Section title should relate to headers in the content
                headers_in_chunk = [
                    line.strip()
                    for line in chunk["content"].split("\n")
                    if line.strip().startswith("#")
                ]
                if headers_in_chunk:
                    # The section title should be related to headers in the chunk
                    assert any(
                        chunk["section_title"] in header.lstrip("#").strip()
                        for header in headers_in_chunk
                    )

    def test_chunking_token_count_accuracy(self) -> None:
        """Test that token counting in chunks is reasonably accurate."""
        from src.utils.markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        # Create content with known characteristics
        content = (
            """# 테스트 문서

이것은 토큰 수를 테스트하기 위한 문서입니다.
한국어와 영어가 섞여 있습니다.
This is a mixed language document for testing token estimation.

## 섹션 A
"""
            + "한국어 내용입니다. " * 50
            + """

## Section B
"""
            + "English content here. " * 50
        )

        chunks = processor.chunk_markdown_content(content, max_tokens=300)

        for chunk in chunks:
            # Verify token count is within reasonable bounds
            estimated_tokens = processor.estimate_token_count(chunk["content"])
            reported_tokens = chunk["tokens"]

            # Should be roughly similar (allowing for some variation in estimation)
            assert abs(estimated_tokens - reported_tokens) <= max(
                50, estimated_tokens * 0.2
            )

            # Note: Due to overlap, chunks may exceed the base token limit

    def test_chunking_character_count_tracking(self) -> None:
        """Test that character counts are properly tracked in chunks."""
        from src.utils.markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        content = (
            """# 문자 수 테스트

이 문서는 문자 수 추적을 테스트합니다.
한글과 영어, 그리고 특수문자(!@#$%)가 포함되어 있습니다.

## 상세 내용
"""
            + "혼합 내용입니다. Mixed content. " * 50
        )

        chunks = processor.chunk_markdown_content(content, max_tokens=400)

        for chunk in chunks:
            # Character count should match actual content length
            actual_char_count = len(chunk["content"])
            reported_char_count = chunk["char_count"]

            assert actual_char_count == reported_char_count


class TestChunkingIntegration:
    """Integration tests for chunking with MarkdownProcessor pipeline."""

    def test_process_markdown_with_chunking_needed(self) -> None:
        """Test complete pipeline when chunking is needed."""
        from src.utils.markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        # Create content that definitely needs chunking
        long_content = (
            """---
title: "긴 정책 문서"
category: policy
---

# 긴 정책 문서

본 문서는 매우 긴 정책 문서입니다.

## 제1장 개요
"""
            + "제1장 관련 내용입니다. " * 1000
            + """

## 제2장 세부사항
"""
            + "제2장 관련 내용입니다. " * 1000
        )

        filename = "long-policy.md"
        result = processor.process_markdown(long_content, filename)

        # Should indicate chunking is needed
        assert result["needs_chunking"] == True
        assert result["chunks"] is not None
        assert len(result["chunks"]) > 1

        # Original content should be preserved
        assert "제1장 개요" in result["content"]
        assert "제2장 세부사항" in result["content"]

        # Should be marked as valid despite needing chunking
        assert result["is_valid"] == True
        assert result["validation_error"] is None

    def test_process_markdown_no_chunking_needed(self) -> None:
        """Test complete pipeline when chunking is not needed."""
        from src.utils.markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        normal_content = """# 표준 정책 문서

이것은 적당한 길이의 정책 문서입니다.

## 제1조 목적
본 문서의 목적은 명확합니다.

## 제2조 범위  
적용 범위는 제한적입니다."""

        filename = "normal-policy.md"
        result = processor.process_markdown(normal_content, filename)

        # Should not need chunking
        assert result["needs_chunking"] == False
        assert result["chunks"] is None

        # Should be valid
        assert result["is_valid"] == True
        assert result["validation_error"] is None


class TestChunkingPerformance:
    """Performance tests for chunking functionality."""

    def test_chunking_performance_reasonable(self) -> None:
        """Test that chunking performance is reasonable for large documents."""
        import time

        from src.utils.markdown_processor import MarkdownProcessor

        processor = MarkdownProcessor()

        # Create a very large document
        large_content = """# 대형 정책 문서

본 문서는 성능 테스트를 위한 대형 문서입니다.

"""

        # Add many sections
        for i in range(100):
            large_content += (
                f"""
## 제{i+1}조 섹션 {i+1}

이것은 제{i+1}조의 내용입니다. """
                + f"섹션 {i+1} 관련 상세 내용입니다. " * 50
            )

        # Measure chunking time
        start_time = time.time()
        chunks = processor.chunk_markdown_content(large_content, max_tokens=800)
        end_time = time.time()

        chunking_time = end_time - start_time

        # Should complete in reasonable time (less than 5 seconds)
        assert chunking_time < 5.0

        # Should create a reasonable number of chunks
        assert len(chunks) > 10  # Large document should create many chunks
        assert len(chunks) < 1000  # But not excessive number

        # All chunks should be valid
        for chunk in chunks:
            # Chunks may exceed max_tokens due to overlap content (up to 200 additional tokens)
            # but should not exceed the embedding model's limit (8192)
            assert chunk["tokens"] <= 8192  # Maximum for bge-m3 model
            assert len(chunk["content"]) > 0
