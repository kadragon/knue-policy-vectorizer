"""Markdown preprocessing for policy documents."""

import re
import hashlib
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

import frontmatter

from logger import setup_logger


class MarkdownProcessor:
    """Processes markdown documents for vectorization."""
    
    def __init__(self):
        """Initialize the MarkdownProcessor."""
        self.logger = setup_logger("INFO", "MarkdownProcessor")
        
        # Token estimation (rough approximation: 1 token ≈ 4 characters for Korean)
        self.chars_per_token = 4
        
        # Content limits
        self.max_chars = 30000  # Maximum character count
        self.max_tokens = 8192  # Maximum token count for bge-m3 model
    
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
                self.logger.debug("YAML frontmatter removed", 
                                metadata_keys=list(post.metadata.keys()))
                return clean_content
                
        except Exception as e:
            self.logger.debug("Failed to parse YAML frontmatter", error=str(e))
        
        # Try manual TOML frontmatter removal
        try:
            # Check for TOML frontmatter (+++...+++)
            if content.startswith('+++'):
                lines = content.split('\n')
                end_idx = -1
                
                # Find the closing +++
                for i, line in enumerate(lines[1:], 1):
                    if line.strip() == '+++':
                        end_idx = i
                        break
                
                if end_idx > 0:
                    # Remove frontmatter and return remaining content
                    remaining_lines = lines[end_idx + 1:]
                    clean_content = '\n'.join(remaining_lines).lstrip('\n')
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
        lines = content.split('\n')
        h1_matches = []
        
        for line in lines:
            if line.startswith('#') and not line.startswith('##'):
                # Extract content after #
                h1_content = line[1:].strip()
                if h1_content:  # Only add non-empty titles
                    h1_matches.append(h1_content)
                    break  # Take first H1 only
        
        if h1_matches:
            # Get first H1, clean formatting
            title = h1_matches[0].strip()
            # Remove markdown formatting
            title = re.sub(r'\*\*(.+?)\*\*', r'\1', title)  # Bold
            title = re.sub(r'\*(.+?)\*', r'\1', title)      # Italic
            title = re.sub(r'`(.+?)`', r'\1', title)        # Code
            title = title.strip()
            
            if title:  # Only return if non-empty after cleaning
                self.logger.debug("Title extracted from H1", title=title)
                return title
        
        # Fallback to filename
        if filename:
            title = Path(filename).stem  # Remove extension
            self.logger.debug("Title extracted from filename", title=title, filename=filename)
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
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Remove trailing whitespace from each line
        lines = []
        for line in content.split('\n'):
            lines.append(line.rstrip())
        
        content = '\n'.join(lines)
        
        # Ensure content doesn't end with multiple newlines
        content = content.rstrip('\n')
        
        self.logger.debug("Content cleaned", 
                         original_length=len(content),
                         final_length=len(content))
        
        return content
    
    def calculate_document_id(self, file_path: str) -> str:
        """Calculate consistent document ID from file path.
        
        Args:
            file_path: Relative file path in repository
            
        Returns:
            Document ID (hex hash)
        """
        # Use MD5 hash of file path for consistent ID
        doc_id = hashlib.md5(file_path.encode('utf-8')).hexdigest()
        
        self.logger.debug("Document ID calculated", 
                         file_path=file_path, 
                         doc_id=doc_id)
        
        return doc_id
    
    def estimate_token_count(self, content: str) -> int:
        """Estimate token count for content.
        
        Args:
            content: Text content
            
        Returns:
            Estimated token count
        """
        # Simple estimation: characters / chars_per_token
        char_count = len(content)
        token_estimate = max(1, char_count // self.chars_per_token)
        
        self.logger.debug("Token count estimated", 
                         char_count=char_count, 
                         estimated_tokens=token_estimate)
        
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
            message = f"Content too long: {char_count} characters (max: {self.max_chars})"
            self.logger.warning("Content length validation failed", 
                              char_count=char_count, 
                              max_chars=self.max_chars)
            return False, message
        
        # Check token limit
        if token_estimate > self.max_tokens:
            message = f"Content too long: ~{token_estimate} tokens (max: {self.max_tokens})"
            self.logger.warning("Token count validation failed", 
                              estimated_tokens=token_estimate, 
                              max_tokens=self.max_tokens)
            return False, message
        
        self.logger.debug("Content length validation passed", 
                         char_count=char_count, 
                         estimated_tokens=token_estimate)
        
        return True, None
    
    def generate_metadata(self, 
                         content: str,
                         title: str, 
                         filename: str,
                         file_path: str,
                         commit_info: Dict[str, str],
                         github_url: str) -> Dict[str, Any]:
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
        
        metadata = {
            'doc_id': doc_id,
            '규정명': title,
            '파일경로': file_path,
            '갱신날짜': commit_info.get('commit_date', ''),
            '업로드시각': upload_time,
            'commit_sha': commit_info.get('commit_sha', ''),
            'source_url': github_url
        }
        
        self.logger.debug("Metadata generated", 
                         doc_id=doc_id, 
                         title=title,
                         file_path=file_path)
        
        return metadata
    
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
            
            # Step 4: Validate length
            is_valid, error_message = self.validate_content_length(clean_content)
            if not is_valid:
                self.logger.error("Content validation failed", 
                                filename=filename, 
                                error=error_message)
                # Still return the processed content, but mark it as invalid
            
            result = {
                'content': clean_content,
                'title': title,
                'filename': filename,
                'is_valid': is_valid,
                'validation_error': error_message,
                'char_count': len(clean_content),
                'estimated_tokens': self.estimate_token_count(clean_content)
            }
            
            self.logger.info("Markdown processing completed", 
                           filename=filename,
                           title=title,
                           char_count=result['char_count'],
                           estimated_tokens=result['estimated_tokens'],
                           is_valid=is_valid)
            
            return result
            
        except Exception as e:
            self.logger.error("Failed to process markdown", 
                            filename=filename, 
                            error=str(e))
            raise
    
    def create_document_for_vectorization(self,
                                        processed_content: Dict[str, Any],
                                        file_path: str,
                                        commit_info: Dict[str, str],
                                        github_url: str) -> Dict[str, Any]:
        """Create final document structure for vectorization.
        
        Args:
            processed_content: Result from process_markdown()
            file_path: Relative file path in repository
            commit_info: Git commit information
            github_url: GitHub URL for the file
            
        Returns:
            Complete document with content and metadata
        """
        if not processed_content['is_valid']:
            self.logger.warning("Creating document from invalid content", 
                              file_path=file_path,
                              error=processed_content['validation_error'])
        
        # Generate metadata
        metadata = self.generate_metadata(
            content=processed_content['content'],
            title=processed_content['title'],
            filename=processed_content['filename'],
            file_path=file_path,
            commit_info=commit_info,
            github_url=github_url
        )
        
        document = {
            'content': processed_content['content'],
            'metadata': metadata,
            'processing_info': {
                'char_count': processed_content['char_count'],
                'estimated_tokens': processed_content['estimated_tokens'],
                'is_valid': processed_content['is_valid'],
                'validation_error': processed_content.get('validation_error')
            }
        }
        
        self.logger.info("Document created for vectorization",
                        doc_id=metadata['doc_id'],
                        title=metadata['규정명'],
                        is_valid=processed_content['is_valid'])
        
        return document