"""Markdown preprocessing for policy documents."""

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import frontmatter

from logger import setup_logger

# Support both package and standalone imports
try:
    from .crypto_utils import CryptoUtils
except Exception:  # pragma: no cover - fallback when imported as a script
    from crypto_utils import CryptoUtils  # type: ignore


class MarkdownProcessor:
    """Processes markdown documents for vectorization."""

    def __init__(self, config=None):
        """Initialize the MarkdownProcessor."""
        # Use config-provided log level when available
        if config is None:
            from config import Config

            config = Config()
        self.logger = setup_logger(
            getattr(config, "log_level", "INFO"), "MarkdownProcessor"
        )

        # Token estimation (conservative approximation: 1 token ≈ 2 characters for Korean)
        # This is more conservative to ensure we chunk documents that might exceed limits
        self.chars_per_token = 2

        # Use provided config or create a default one
        # config already ensured above

        # Content limits
        self.max_chars = config.max_document_chars
        self.max_tokens = config.max_tokens
        self.chunk_threshold = config.chunk_threshold
        self.chunk_overlap = config.chunk_overlap

    def remove_frontmatter(self, content: str) -> str:
        """Remove YAML or TOML frontmatter from markdown content.

        Args:
            content: Raw markdown content potentially with frontmatter

        Returns:
            Markdown content without frontmatter
        """
        try:
            # Try using python-frontmatter first (handles YAML)
            post = frontmatter.loads(content)

            # If metadata was parsed, return content without frontmatter
            if post.metadata:
                clean_content = post.content
                self.logger.debug(
                    "YAML frontmatter removed", metadata_keys=list(post.metadata.keys())
                )
                return clean_content

        except Exception as e:
            self.logger.debug("Failed to parse YAML frontmatter", error=str(e))

        # Try manual TOML frontmatter removal
        try:
            # Check for TOML frontmatter (+++...+++)
            if content.startswith("+++"):
                lines = content.split("\n")
                end_idx = -1

                # Find the closing +++
                for i, line in enumerate(lines[1:], 1):
                    if line.strip() == "+++":
                        end_idx = i
                        break

                if end_idx > 0:
                    # Remove frontmatter and return remaining content
                    remaining_lines = lines[end_idx + 1 :]
                    clean_content = "\n".join(remaining_lines).lstrip("\n")
                    self.logger.debug("TOML frontmatter removed")
                    return clean_content

        except Exception as e:
            self.logger.debug("Failed to parse TOML frontmatter", error=str(e))

        # If no frontmatter found or parsing failed, return original content
        self.logger.debug("No frontmatter detected")
        return content

    def extract_title(self, content: str, filename: str = "") -> str:
        """Extract title from markdown content or filename.

        Args:
            content: Markdown content
            filename: Original filename (used as fallback)

        Returns:
            Extracted title
        """
        # Try to extract from H1 heading first
        # Split into lines and find H1 lines specifically
        lines = content.split("\n")
        h1_matches = []

        for line in lines:
            if line.startswith("#") and not line.startswith("##"):
                # Extract content after #
                h1_content = line[1:].strip()
                if h1_content:  # Only add non-empty titles
                    h1_matches.append(h1_content)
                    break  # Take first H1 only

        if h1_matches:
            # Get first H1, clean formatting
            title = h1_matches[0].strip()
            # Remove markdown formatting
            title = re.sub(r"\*\*(.+?)\*\*", r"\1", title)  # Bold
            title = re.sub(r"\*(.+?)\*", r"\1", title)  # Italic
            title = re.sub(r"`(.+?)`", r"\1", title)  # Code
            title = title.strip()

            if title:  # Only return if non-empty after cleaning
                self.logger.debug("Title extracted from H1", title=title)
                return title

        # Fallback to filename
        if filename:
            title = Path(filename).stem  # Remove extension
            self.logger.debug(
                "Title extracted from filename", title=title, filename=filename
            )
            return title

        # Last resort fallback
        self.logger.warning("No title found, using default")
        return "Untitled Document"

    def clean_content(self, content: str) -> str:
        """Clean markdown content by removing excessive whitespace.

        Args:
            content: Raw markdown content

        Returns:
            Cleaned markdown content
        """
        # Remove leading and trailing whitespace
        content = content.strip()

        # Replace multiple consecutive newlines with double newlines
        content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)

        # Remove trailing whitespace from each line
        lines = []
        for line in content.split("\n"):
            lines.append(line.rstrip())

        content = "\n".join(lines)

        # Ensure content doesn't end with multiple newlines
        content = content.rstrip("\n")

        self.logger.debug(
            "Content cleaned", original_length=len(content), final_length=len(content)
        )

        return content

    def calculate_document_id(self, file_path: str) -> str:
        """Calculate consistent document ID from file path.

        Args:
            file_path: Relative file path in repository

        Returns:
            Document ID (hex hash)
        """
        # Use SHA-256 hash of file path for consistent ID (data integrity)
        doc_id = CryptoUtils.calculate_data_integrity_hash(file_path)

        self.logger.debug("Document ID calculated", file_path=file_path, doc_id=doc_id)

        return doc_id

    def estimate_token_count(
        self, content: str, use_embedding_service: bool = True
    ) -> int:
        """Estimate token count for content.

        Args:
            content: Text content
            use_embedding_service: If True, use embedding service's tokenizer for accuracy

        Returns:
            Estimated token count
        """
        if use_embedding_service:
            try:
                # Import here to avoid circular imports
                from embedding_service import EmbeddingService

                # Create a temporary embedding service for tokenization
                if not hasattr(self, "_embedding_service"):
                    self._embedding_service = EmbeddingService()

                # Use the embedding service's tokenizer for accurate counting
                token_estimate = self._embedding_service.estimate_tokens(content)

                self.logger.debug(
                    "Token count estimated using embedding service",
                    char_count=len(content),
                    estimated_tokens=token_estimate,
                )

                return token_estimate

            except Exception as e:
                self.logger.warning(
                    "Failed to use embedding service for token estimation, falling back to simple method",
                    error=str(e),
                )
                # Fall back to simple estimation

        # Simple estimation: characters / chars_per_token
        char_count = len(content)
        token_estimate = max(1, char_count // self.chars_per_token)

        self.logger.debug(
            "Token count estimated using character method",
            char_count=char_count,
            estimated_tokens=token_estimate,
        )

        return token_estimate

    def validate_content_length(self, content: str) -> Tuple[bool, Optional[str]]:
        """Validate content length against limits.

        Args:
            content: Text content to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        char_count = len(content)
        token_estimate = self.estimate_token_count(content)

        # Check character limit
        if char_count > self.max_chars:
            message = (
                f"Content too long: {char_count} characters (max: {self.max_chars})"
            )
            self.logger.warning(
                "Content length validation failed",
                char_count=char_count,
                max_chars=self.max_chars,
            )
            return False, message

        # Check token limit (only fail if exceeding embedding service limit)
        if token_estimate > self.max_tokens:
            message = (
                f"Content too long: ~{token_estimate} tokens (max: {self.max_tokens})"
            )
            self.logger.warning(
                "Token count validation failed",
                estimated_tokens=token_estimate,
                max_tokens=self.max_tokens,
            )
            return False, message

        self.logger.debug(
            "Content length validation passed",
            char_count=char_count,
            estimated_tokens=token_estimate,
        )

        return True, None

    def generate_metadata(
        self,
        content: str,
        title: str,
        filename: str,
        file_path: str,
        commit_info: Dict[str, str],
        github_url: str,
    ) -> Dict[str, Any]:
        """Generate metadata for processed document.

        Args:
            content: Processed markdown content
            title: Extracted title
            filename: Original filename
            file_path: Relative file path in repository
            commit_info: Git commit information
            github_url: GitHub URL for the file

        Returns:
            Metadata dictionary
        """
        doc_id = self.calculate_document_id(file_path)

        # Generate upload timestamp
        upload_time = datetime.now().isoformat()

        # Calculate content metrics
        content_length = len(content)
        estimated_tokens = self.estimate_token_count(content)

        metadata = {
            "document_id": doc_id,
            "title": title,  # Use the actual Korean title
            "file_path": file_path,
            "last_modified": upload_time,
            "commit_hash": commit_info.get("sha", ""),
            "github_url": github_url,
            "content_length": content_length,
            "estimated_tokens": estimated_tokens,
            "content": content,  # Add the actual content
            # Korean-specific fields (non-duplicated)
            "korean_title": title,  # Keep for consistency/backward compatibility
        }

        self.logger.debug(
            "Metadata generated", doc_id=doc_id, title=title, file_path=file_path
        )

        return metadata

    def _split_oversized_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a chunk that exceeds the embedding service token limit.

        Args:
            chunk: Oversized chunk to split

        Returns:
            List of smaller chunks that fit within token limits
        """
        content = chunk["content"]
        section_title = chunk["section_title"]

        # Use a conservative target to ensure we stay under limits
        target_tokens = min(self.max_tokens - 100, 7000)  # Leave buffer for safety

        # Split content into sentences/paragraphs for better splitting points
        lines = content.split("\n")
        sub_chunks = []
        current_sub_chunk = []
        current_tokens = 0
        sub_chunk_index = 0

        for line in lines:
            line_tokens = self.estimate_token_count(line)

            # If adding this line would exceed limit, create a sub-chunk
            if current_tokens + line_tokens > target_tokens and current_sub_chunk:
                sub_chunk_content = "\n".join(current_sub_chunk)
                sub_chunk = {
                    "content": sub_chunk_content,
                    "tokens": self.estimate_token_count(sub_chunk_content),
                    "chunk_index": chunk["chunk_index"] + sub_chunk_index,
                    "section_title": section_title,
                    "char_count": len(sub_chunk_content),
                    "overlap_tokens": 0,
                    "has_context_overlap": False,
                    "start_line": -1,  # Not tracking lines for sub-chunks
                    "end_line": -1,
                }
                sub_chunks.append(sub_chunk)
                current_sub_chunk = []
                current_tokens = 0
                sub_chunk_index += 1

            # Handle extremely long single lines by splitting them further
            if line_tokens > target_tokens:
                # Split long line by sentences or words
                words = line.split(" ")
                word_chunk = []
                word_chunk_tokens = 0

                for word in words:
                    word_tokens = self.estimate_token_count(word)
                    if word_chunk_tokens + word_tokens > target_tokens and word_chunk:
                        # Create sub-chunk from words
                        word_content = " ".join(word_chunk)
                        if word_content.strip():
                            sub_chunk = {
                                "content": word_content,
                                "tokens": self.estimate_token_count(word_content),
                                "chunk_index": chunk["chunk_index"] + sub_chunk_index,
                                "section_title": section_title,
                                "char_count": len(word_content),
                                "overlap_tokens": 0,
                                "has_context_overlap": False,
                                "start_line": -1,
                                "end_line": -1,
                            }
                            sub_chunks.append(sub_chunk)
                            sub_chunk_index += 1
                        word_chunk = [word]
                        word_chunk_tokens = word_tokens
                    else:
                        word_chunk.append(word)
                        word_chunk_tokens += word_tokens

                # Add remaining words as a chunk
                if word_chunk:
                    word_content = " ".join(word_chunk)
                    if word_content.strip():
                        sub_chunk = {
                            "content": word_content,
                            "tokens": self.estimate_token_count(word_content),
                            "chunk_index": chunk["chunk_index"] + sub_chunk_index,
                            "section_title": section_title,
                            "char_count": len(word_content),
                            "overlap_tokens": 0,
                            "has_context_overlap": False,
                            "start_line": -1,
                            "end_line": -1,
                        }
                        sub_chunks.append(sub_chunk)
                        sub_chunk_index += 1
            else:
                current_sub_chunk.append(line)
                current_tokens += line_tokens

        # Add the last sub-chunk if it has content
        if current_sub_chunk:
            sub_chunk_content = "\n".join(current_sub_chunk)
            sub_chunk = {
                "content": sub_chunk_content,
                "tokens": self.estimate_token_count(sub_chunk_content),
                "chunk_index": chunk["chunk_index"] + sub_chunk_index,
                "section_title": section_title,
                "char_count": len(sub_chunk_content),
                "overlap_tokens": 0,
                "has_context_overlap": False,
                "start_line": -1,
                "end_line": -1,
            }
            sub_chunks.append(sub_chunk)

        self.logger.info(
            "Split oversized chunk",
            original_tokens=chunk["tokens"],
            sub_chunks=len(sub_chunks),
            avg_tokens=(
                sum(c["tokens"] for c in sub_chunks) // len(sub_chunks)
                if sub_chunks
                else 0
            ),
        )

        return sub_chunks

    def chunk_markdown_content(
        self, content: str, max_tokens: int = None
    ) -> List[Dict[str, Any]]:
        """Intelligently chunk markdown content with configurable chunking strategy.

        This implements a sophisticated chunking strategy:
        - Primary chunks: up to chunk_threshold tokens (default 800)
        - Nested overlap: chunk_overlap tokens from previous chunk for context continuity
        - Respects markdown structure (headers, lists, code blocks)
        - Maintains semantic boundaries

        Args:
            content: Markdown content to chunk
            max_tokens: Maximum tokens per primary chunk (defaults to self.chunk_threshold)

        Returns:
            List of chunks with metadata and overlap information
        """
        if max_tokens is None:
            max_tokens = self.chunk_threshold  # Use configured chunk threshold

        overlap_tokens = self.chunk_overlap  # Use configured overlap size

        if not content.strip():
            return []

        chunks = []
        lines = content.split("\n")

        # Track document structure
        headers = []
        code_blocks = []
        current_section = ""

        # Parse document structure first
        in_code_block = False
        code_block_start = -1

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Track code blocks
            if stripped.startswith("```"):
                if not in_code_block:
                    in_code_block = True
                    code_block_start = i
                else:
                    code_blocks.append((code_block_start, i))
                    in_code_block = False

            # Track headers
            if stripped.startswith("#") and not in_code_block:
                level = len(stripped) - len(stripped.lstrip("#"))
                title = stripped.lstrip("#").strip()
                headers.append(
                    {"line": i, "level": level, "title": title, "full_line": line}
                )

        # Create chunks with intelligent splitting
        chunk_index = 0
        current_line = 0
        previous_chunk_overlap = []  # Lines from previous chunk for context

        def create_chunk(
            start_line: int, end_line: int, include_overlap: bool = True
        ) -> Dict[str, Any]:
            nonlocal chunk_index, previous_chunk_overlap

            # Determine chunk boundaries
            chunk_lines = []

            # Add overlap from previous chunk if requested and available
            if include_overlap and previous_chunk_overlap and chunk_index > 0:
                chunk_lines.extend(previous_chunk_overlap)

            # Add main content
            chunk_lines.extend(lines[start_line:end_line])

            chunk_content = "\n".join(chunk_lines)

            # Find current section for this chunk
            section_title = ""
            for header in reversed(headers):
                if header["line"] < end_line:
                    section_title = header["title"]
                    break

            # Calculate metrics (exclude overlap from token limit validation)
            main_content = "\n".join(lines[start_line:end_line])
            main_tokens = self.estimate_token_count(main_content)

            # Full chunk tokens for reporting
            tokens = self.estimate_token_count(chunk_content)
            char_count = len(chunk_content)

            # Prepare overlap for next chunk (last 200 tokens worth of content)
            overlap_lines = []
            if end_line < len(lines):  # Not the last chunk
                # Calculate lines for ~200 tokens of overlap
                overlap_token_count = 0
                for line_idx in range(end_line - 1, max(start_line, end_line - 20), -1):
                    if line_idx >= 0:
                        line_tokens = self.estimate_token_count(lines[line_idx])
                        if overlap_token_count + line_tokens <= overlap_tokens:
                            overlap_lines.insert(0, lines[line_idx])
                            overlap_token_count += line_tokens
                        else:
                            break

                previous_chunk_overlap = overlap_lines

            # Check if we actually added overlap content to this chunk
            actual_overlap = (
                include_overlap
                and chunk_index > 0
                and previous_chunk_overlap
                and len(previous_chunk_overlap) > 0
            )

            chunk = {
                "content": chunk_content,
                "tokens": tokens,
                "chunk_index": chunk_index,
                "section_title": section_title,
                "char_count": char_count,
                "overlap_tokens": len(previous_chunk_overlap) if actual_overlap else 0,
                "has_context_overlap": actual_overlap,
                "start_line": start_line,
                "end_line": end_line,
            }

            chunk_index += 1
            return chunk

        # Main chunking logic
        while current_line < len(lines):
            chunk_start = current_line
            main_content_tokens = 0  # Only count main content tokens, not overlap

            # Find optimal chunk end point
            best_break_point = current_line
            last_good_break = current_line

            # Scan forward to find good breaking points
            for line_idx in range(current_line, len(lines)):
                line = lines[line_idx]
                line_tokens = self.estimate_token_count(line)

                # Check if adding this line would exceed limit (only counting main content)
                if main_content_tokens + line_tokens > max_tokens:
                    if line_idx > current_line:  # We have some content
                        best_break_point = line_idx
                        break
                    else:  # Single line is too long, include it anyway
                        best_break_point = line_idx + 1
                        break

                main_content_tokens += line_tokens

                # Track good breaking points (after headers, empty lines, etc.)
                stripped = line.strip()
                if not stripped:  # Empty line - good break point
                    last_good_break = line_idx + 1
                elif stripped.startswith("#"):  # Header - good break point after
                    last_good_break = line_idx + 1
                elif any(
                    stripped.startswith(marker) for marker in ["---", "***", "___"]
                ):  # Horizontal rule
                    last_good_break = line_idx + 1
                elif line_idx + 1 < len(lines) and lines[
                    line_idx + 1
                ].strip().startswith("#"):
                    # Next line is header - good break point here
                    last_good_break = line_idx + 1

                # Check for code block boundaries
                for start, end in code_blocks:
                    if start <= line_idx <= end:
                        # Don't break inside code blocks
                        if line_idx == end:  # End of code block - good break point
                            last_good_break = line_idx + 1
                        break
                else:
                    # Not in code block
                    if line_idx == len(lines) - 1:  # Last line
                        best_break_point = len(lines)
                        break
            else:
                # Reached end of document
                best_break_point = len(lines)

            # Use the better break point if it's reasonable
            if (
                last_good_break > current_line
                and last_good_break <= best_break_point
                and best_break_point - current_line > 5
            ):  # Don't use if chunk would be tiny
                best_break_point = last_good_break

            # Ensure we make progress
            if best_break_point <= current_line:
                best_break_point = min(current_line + 1, len(lines))

            # Create chunk
            chunk = create_chunk(current_line, best_break_point)
            chunks.append(chunk)

            # Move to next chunk
            current_line = best_break_point

        # Handle edge case: no chunks created for very short content
        if not chunks and content.strip():
            chunk = {
                "content": content,
                "tokens": self.estimate_token_count(content),
                "chunk_index": 0,
                "section_title": "",
                "char_count": len(content),
                "overlap_tokens": 0,
                "has_context_overlap": False,
                "start_line": 0,
                "end_line": len(lines),
            }
            chunks.append(chunk)

        # Validate and fix chunks that exceed embedding service token limits
        validated_chunks = []
        for chunk in chunks:
            if chunk["tokens"] > self.max_tokens:
                self.logger.warning(
                    "Chunk exceeds embedding token limit, splitting further",
                    chunk_index=chunk["chunk_index"],
                    chunk_tokens=chunk["tokens"],
                    max_tokens=self.max_tokens,
                )
                # Split oversized chunk into smaller pieces
                sub_chunks = self._split_oversized_chunk(chunk)
                validated_chunks.extend(sub_chunks)
            else:
                validated_chunks.append(chunk)

        # Re-index chunks after potential splitting
        for i, chunk in enumerate(validated_chunks):
            chunk["chunk_index"] = i

        # Log results
        total_original_tokens = self.estimate_token_count(content)
        total_chunk_tokens = sum(chunk["tokens"] for chunk in validated_chunks)

        self.logger.info(
            "Content chunked with 800/200 strategy",
            total_chunks=len(validated_chunks),
            original_tokens=total_original_tokens,
            total_chunk_tokens=total_chunk_tokens,
            avg_tokens_per_chunk=(
                total_chunk_tokens // len(validated_chunks) if validated_chunks else 0
            ),
            chunks_with_overlap=sum(
                1 for c in validated_chunks if c["has_context_overlap"]
            ),
        )

        return validated_chunks

    def process_markdown(self, raw_content: str, filename: str) -> Dict[str, Any]:
        """Process markdown document through complete pipeline.

        Args:
            raw_content: Raw markdown content
            filename: Original filename

        Returns:
            Dictionary with processed content and extracted information
        """
        self.logger.info("Processing markdown document", filename=filename)

        try:
            # Step 1: Remove frontmatter
            content_no_frontmatter = self.remove_frontmatter(raw_content)

            # Step 2: Extract title
            title = self.extract_title(content_no_frontmatter, filename)

            # Step 3: Clean content
            clean_content = self.clean_content(content_no_frontmatter)

            # Step 4: Check if content needs chunking (use chunk_threshold, not max_tokens)
            estimated_tokens = self.estimate_token_count(clean_content)

            if estimated_tokens > self.chunk_threshold:
                # Content is too long, chunk it
                chunks = self.chunk_markdown_content(clean_content)
                self.logger.info(
                    "Document chunked due to length",
                    filename=filename,
                    original_tokens=estimated_tokens,
                    chunk_count=len(chunks),
                )

                result = {
                    "content": clean_content,  # Keep original for reference
                    "title": title,
                    "filename": filename,
                    "is_valid": True,  # Chunked content is valid
                    "validation_error": None,
                    "char_count": len(clean_content),
                    "estimated_tokens": estimated_tokens,
                    "needs_chunking": True,
                    "chunks": chunks,
                }
            else:
                # Content is within limits
                is_valid, error_message = self.validate_content_length(clean_content)
                if not is_valid:
                    self.logger.error(
                        "Content validation failed",
                        filename=filename,
                        error=error_message,
                    )

                result = {
                    "content": clean_content,
                    "title": title,
                    "filename": filename,
                    "is_valid": is_valid,
                    "validation_error": error_message,
                    "char_count": len(clean_content),
                    "estimated_tokens": estimated_tokens,
                    "needs_chunking": False,
                    "chunks": None,
                }

            self.logger.info(
                "Markdown processing completed",
                filename=filename,
                title=title,
                char_count=result["char_count"],
                estimated_tokens=result["estimated_tokens"],
                is_valid=result["is_valid"],
            )

            return result

        except Exception as e:
            self.logger.error(
                "Failed to process markdown", filename=filename, error=str(e)
            )
            raise

    def create_document_for_vectorization(
        self,
        processed_content: Dict[str, Any],
        file_path: str,
        commit_info: Dict[str, str],
        github_url: str,
    ) -> List[Dict[str, Any]]:
        """Create final document structure(s) for vectorization.

        For documents that need chunking, this creates multiple documents,
        one for each chunk. For regular documents, returns a single document.

        Args:
            processed_content: Result from process_markdown()
            file_path: Relative file path in repository
            commit_info: Git commit information
            github_url: GitHub URL for the file

        Returns:
            List of complete documents with content and metadata
        """
        if not processed_content["is_valid"]:
            self.logger.warning(
                "Creating document from invalid content",
                file_path=file_path,
                error=processed_content["validation_error"],
            )

        documents = []

        if processed_content["needs_chunking"] and processed_content["chunks"]:
            # Create multiple documents from chunks
            for chunk in processed_content["chunks"]:
                # Generate unique metadata for each chunk
                chunk_metadata = self.generate_metadata(
                    content=chunk["content"],
                    title=f"{processed_content['title']} (Part {chunk['chunk_index'] + 1})",
                    filename=processed_content["filename"],
                    file_path=file_path,
                    commit_info=commit_info,
                    github_url=github_url,
                )

                # Add chunk-specific metadata
                chunk_metadata.update(
                    {
                        "is_chunk": True,
                        "chunk_index": chunk["chunk_index"],
                        "total_chunks": len(processed_content["chunks"]),
                        "section_title": chunk["section_title"],
                        "has_context_overlap": chunk["has_context_overlap"],
                        "overlap_tokens": chunk["overlap_tokens"],
                        "parent_document_id": self.calculate_document_id(file_path),
                    }
                )

                # Modify document ID to be unique for each chunk
                original_id = chunk_metadata["document_id"]
                chunk_metadata["document_id"] = (
                    f"{original_id}_chunk_{chunk['chunk_index']}"
                )

                document = {
                    "content": chunk["content"],
                    "metadata": chunk_metadata,
                    "processing_info": {
                        "char_count": chunk["char_count"],
                        "estimated_tokens": chunk["tokens"],
                        "is_valid": processed_content["is_valid"],
                        "validation_error": processed_content.get("validation_error"),
                        "is_chunk": True,
                        "chunk_info": {
                            "chunk_index": chunk["chunk_index"],
                            "total_chunks": len(processed_content["chunks"]),
                            "section_title": chunk["section_title"],
                            "has_context_overlap": chunk["has_context_overlap"],
                            "overlap_tokens": chunk["overlap_tokens"],
                        },
                    },
                }

                documents.append(document)

            self.logger.info(
                "Created chunked documents for vectorization",
                file_path=file_path,
                total_chunks=len(documents),
                title=processed_content["title"],
            )

        else:
            # Create single document (traditional path)
            metadata = self.generate_metadata(
                content=processed_content["content"],
                title=processed_content["title"],
                filename=processed_content["filename"],
                file_path=file_path,
                commit_info=commit_info,
                github_url=github_url,
            )

            # Add non-chunk metadata
            metadata.update({"is_chunk": False, "chunk_index": 0, "total_chunks": 1})

            document = {
                "content": processed_content["content"],
                "metadata": metadata,
                "processing_info": {
                    "char_count": processed_content["char_count"],
                    "estimated_tokens": processed_content["estimated_tokens"],
                    "is_valid": processed_content["is_valid"],
                    "validation_error": processed_content.get("validation_error"),
                    "is_chunk": False,
                },
            }

            documents.append(document)

            self.logger.info(
                "Created single document for vectorization",
                doc_id=metadata["document_id"],
                title=metadata["title"],
                is_valid=processed_content["is_valid"],
            )

        return documents
