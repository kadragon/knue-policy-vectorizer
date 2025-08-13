"""Tests for the complete sync pipeline (TDD approach)."""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config


class TestSyncPipelineInit:
    """Test SyncPipeline initialization."""
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        from sync_pipeline import SyncPipeline
        
        pipeline = SyncPipeline()
        assert pipeline.config is not None
        assert pipeline.config.repo_url == "https://github.com/kadragon/KNUE-Policy-Hub.git"
        assert pipeline.config.branch == "main"
        assert pipeline.config.qdrant_collection == "knue-policy-idx"
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        from sync_pipeline import SyncPipeline
        
        config = Config(
            repo_url="https://github.com/custom/repo.git",
            branch="develop",
            qdrant_collection="custom_collection"
        )
        
        pipeline = SyncPipeline(config=config)
        assert pipeline.config.repo_url == "https://github.com/custom/repo.git"
        assert pipeline.config.branch == "develop"
        assert pipeline.config.qdrant_collection == "custom_collection"
    
    def test_components_lazy_initialization(self):
        """Test that components are lazily initialized."""
        from sync_pipeline import SyncPipeline
        
        pipeline = SyncPipeline()
        
        # Components should not be initialized until first access
        assert not hasattr(pipeline, '_git_watcher')
        assert not hasattr(pipeline, '_markdown_processor')
        assert not hasattr(pipeline, '_embedding_service')
        assert not hasattr(pipeline, '_qdrant_service')
        
        # Access should trigger initialization
        git_watcher = pipeline.git_watcher
        assert git_watcher is not None
        assert hasattr(pipeline, '_git_watcher')


class TestSyncPipelineHealthChecks:
    """Test health check functionality."""
    
    @patch('sync_pipeline.SyncPipeline.embedding_service')
    @patch('sync_pipeline.SyncPipeline.qdrant_service')
    def test_health_check_success(self, mock_qdrant, mock_embedding):
        """Test successful health check."""
        from sync_pipeline import SyncPipeline
        
        # Mock successful health checks
        mock_embedding.health_check.return_value = True
        mock_qdrant.health_check.return_value = True
        
        pipeline = SyncPipeline()
        result = pipeline.health_check()
        
        assert result is True
        mock_embedding.health_check.assert_called_once()
        mock_qdrant.health_check.assert_called_once()
    
    @patch('sync_pipeline.SyncPipeline.embedding_service')
    @patch('sync_pipeline.SyncPipeline.qdrant_service')
    def test_health_check_embedding_failure(self, mock_qdrant, mock_embedding):
        """Test health check with embedding service failure."""
        from sync_pipeline import SyncPipeline
        
        # Mock embedding failure
        mock_embedding.health_check.return_value = False
        mock_qdrant.health_check.return_value = True
        
        pipeline = SyncPipeline()
        result = pipeline.health_check()
        
        assert result is False
    
    @patch('sync_pipeline.SyncPipeline.embedding_service')
    @patch('sync_pipeline.SyncPipeline.qdrant_service')
    def test_health_check_qdrant_failure(self, mock_qdrant, mock_embedding):
        """Test health check with Qdrant service failure."""
        from sync_pipeline import SyncPipeline
        
        # Mock Qdrant failure
        mock_embedding.health_check.return_value = True
        mock_qdrant.health_check.return_value = False
        
        pipeline = SyncPipeline()
        result = pipeline.health_check()
        
        assert result is False


class TestSyncPipelineCollectionManagement:
    """Test collection management functionality."""
    
    @patch('sync_pipeline.SyncPipeline.qdrant_service')
    def test_ensure_collection_exists_creates_new(self, mock_qdrant):
        """Test collection creation when it doesn't exist."""
        from sync_pipeline import SyncPipeline
        
        # Mock collection doesn't exist
        mock_qdrant.collection_exists.return_value = False
        mock_qdrant.create_collection.return_value = True
        
        pipeline = SyncPipeline()
        result = pipeline._ensure_collection_exists()
        
        assert result is True
        mock_qdrant.collection_exists.assert_called_once_with()
        mock_qdrant.create_collection.assert_called_once()
    
    @patch('sync_pipeline.SyncPipeline.qdrant_service')
    def test_ensure_collection_exists_already_exists(self, mock_qdrant):
        """Test when collection already exists."""
        from sync_pipeline import SyncPipeline
        
        # Mock collection exists
        mock_qdrant.collection_exists.return_value = True
        
        pipeline = SyncPipeline()
        result = pipeline._ensure_collection_exists()
        
        assert result is True
        mock_qdrant.collection_exists.assert_called_once_with()
        mock_qdrant.create_collection.assert_not_called()


class TestSyncPipelineMainSync:
    """Test main synchronization logic."""
    
    @patch('sync_pipeline.SyncPipeline.qdrant_service')
    @patch('sync_pipeline.SyncPipeline.embedding_service')
    @patch('sync_pipeline.SyncPipeline.markdown_processor')
    @patch('sync_pipeline.SyncPipeline.git_watcher')
    def test_sync_no_changes(self, mock_git, mock_md, mock_embedding, mock_qdrant):
        """Test sync when there are no changes."""
        from sync_pipeline import SyncPipeline
        
        # Mock no changes
        mock_git.has_changes.return_value = False
        
        pipeline = SyncPipeline()
        result = pipeline.sync()
        
        assert result['status'] == 'success'
        assert result['changes_detected'] is False
        assert result['upserted'] == 0
        assert result['deleted'] == 0
        
        # Should not process any files
        mock_md.process_markdown.assert_not_called()
        mock_embedding.generate_embedding.assert_not_called()
        mock_qdrant.upsert_point.assert_not_called()
    
    @patch('sync_pipeline.SyncPipeline.qdrant_service')
    @patch('sync_pipeline.SyncPipeline.embedding_service')
    @patch('sync_pipeline.SyncPipeline.markdown_processor')
    @patch('sync_pipeline.SyncPipeline.git_watcher')
    def test_sync_with_added_files(self, mock_git, mock_md, mock_embedding, mock_qdrant):
        """Test sync with newly added markdown files."""
        from sync_pipeline import SyncPipeline
        
        # Mock changes with added files
        mock_git.has_changes.return_value = True
        mock_git.get_changed_files.return_value = (
            ['new_policy.md'],  # added
            [],                 # modified
            []                  # deleted
        )
        
        # Mock file processing
        mock_git.get_file_content.return_value = "# New Policy\nContent here"
        mock_git.get_file_commit_info.return_value = {
            'commit_sha': 'abc123',
            'commit_date': '2024-01-01T00:00:00Z'
        }
        mock_md.process_markdown.return_value = {
            'title': 'New Policy',
            'content': 'Processed content',
            'filename': 'new_policy.md',
            'is_valid': True,
            'validation_error': None,
            'char_count': 100,
            'estimated_tokens': 25
        }
        mock_md.calculate_document_id.return_value = 'doc123'
        mock_md.generate_metadata.return_value = {'category': 'policy'}
        mock_embedding.generate_embedding.return_value = [0.1] * 1024
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.upsert_point.return_value = True
        
        pipeline = SyncPipeline()
        result = pipeline.sync()
        
        assert result['status'] == 'success'
        assert result['changes_detected'] is True
        assert result['upserted'] == 1
        assert result['deleted'] == 0
        assert len(result['processed_files']) == 1
        
        # Verify processing chain
        mock_md.process_markdown.assert_called_once()
        mock_embedding.generate_embedding.assert_called_once()
        mock_qdrant.upsert_point.assert_called_once()
    
    @patch('sync_pipeline.SyncPipeline.qdrant_service')
    @patch('sync_pipeline.SyncPipeline.embedding_service')
    @patch('sync_pipeline.SyncPipeline.markdown_processor')
    @patch('sync_pipeline.SyncPipeline.git_watcher')
    def test_sync_with_deleted_files(self, mock_git, mock_md, mock_embedding, mock_qdrant):
        """Test sync with deleted markdown files."""
        from sync_pipeline import SyncPipeline
        
        # Mock changes with deleted files
        mock_git.has_changes.return_value = True
        mock_git.get_changed_files.return_value = (
            [],                 # added
            [],                 # modified
            ['old_policy.md']   # deleted
        )
        
        # Mock file ID calculation for deleted files
        mock_md.calculate_document_id.return_value = 'doc456'
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.delete_document_chunks.return_value = True
        
        pipeline = SyncPipeline()
        result = pipeline.sync()
        
        assert result['status'] == 'success'
        assert result['changes_detected'] is True
        assert result['upserted'] == 0
        assert result['deleted'] == 1
        assert len(result['deleted_files']) == 1
        
        # Verify deletion
        mock_qdrant.delete_document_chunks.assert_called_once_with('doc456')
        
        # Should not generate embeddings for deleted files
        mock_embedding.generate_embedding.assert_not_called()
    
    @patch('sync_pipeline.SyncPipeline.qdrant_service')
    @patch('sync_pipeline.SyncPipeline.embedding_service')
    @patch('sync_pipeline.SyncPipeline.markdown_processor')
    @patch('sync_pipeline.SyncPipeline.git_watcher')
    def test_sync_with_mixed_changes(self, mock_git, mock_md, mock_embedding, mock_qdrant):
        """Test sync with added, modified, and deleted files."""
        from sync_pipeline import SyncPipeline
        
        # Mock mixed changes
        mock_git.has_changes.return_value = True
        mock_git.get_changed_files.return_value = (
            ['new_policy.md'],      # added
            ['updated_policy.md'],  # modified
            ['old_policy.md']       # deleted
        )
        
        # Mock file content for added/modified files
        def mock_get_file_content(file_path):
            return f"# Content for {file_path}\nSome content here"
        
        mock_git.get_file_content.side_effect = mock_get_file_content
        
        # Mock file commit info
        def mock_get_file_commit_info(file_path):
            return {
                'commit_sha': f'sha_for_{file_path}',
                'commit_date': '2024-01-01T00:00:00Z'
            }
        
        mock_git.get_file_commit_info.side_effect = mock_get_file_commit_info
        
        # Mock processing results
        def mock_process_markdown(content, file_path):
            return {
                'title': f'Title for {file_path}',
                'content': f'Processed {content}',
                'filename': file_path,
                'is_valid': True,
                'validation_error': None,
                'char_count': len(content),
                'estimated_tokens': len(content) // 4
            }
        
        def mock_calculate_document_id(file_path):
            return f'doc_{file_path.replace(".md", "")}'
        
        def mock_generate_metadata(content, title, filename, file_path, commit_info, github_url):
            return {
                'document_id': f'doc_{filename.replace(".md", "")}',
                'title': title,
                'file_path': file_path,
                'last_modified': commit_info.get('commit_date', '2024-01-01T00:00:00Z'),
                'commit_hash': commit_info.get('commit_sha', 'abc123'),
                'github_url': github_url,
                'content_length': len(content),
                'estimated_tokens': len(content) // 4,
                'content': content,
                'chunk_index': 0,
                'total_chunks': 1,
                'section_title': title,
                'chunk_tokens': len(content) // 4,
                'is_chunk': False
            }
        
        mock_md.process_markdown.side_effect = mock_process_markdown
        mock_md.calculate_document_id.side_effect = mock_calculate_document_id
        mock_md.generate_metadata.side_effect = mock_generate_metadata
        mock_md.calculate_document_id.return_value = 'doc_old_policy'
        mock_embedding.generate_embedding.return_value = [0.1] * 1024
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.upsert_point.return_value = True
        mock_qdrant.delete_document_chunks.return_value = True
        
        pipeline = SyncPipeline()
        result = pipeline.sync()
        
        assert result['status'] == 'success'
        assert result['changes_detected'] is True
        assert result['upserted'] == 2  # new + modified
        assert result['deleted'] == 1   # deleted
        assert len(result['processed_files']) == 2
        assert len(result['deleted_files']) == 1
        
        # Verify processing
        assert mock_md.process_markdown.call_count == 2
        assert mock_embedding.generate_embedding.call_count == 2
        assert mock_qdrant.upsert_point.call_count == 2
        assert mock_qdrant.delete_document_chunks.call_count == 1


class TestSyncPipelineErrorHandling:
    """Test error handling in sync pipeline."""
    
    @patch('sync_pipeline.SyncPipeline.git_watcher')
    def test_sync_git_error(self, mock_git):
        """Test handling of Git-related errors."""
        from sync_pipeline import SyncPipeline, SyncError
        
        # Mock Git error
        mock_git.pull_updates.side_effect = Exception("Git pull failed")
        
        pipeline = SyncPipeline()
        
        with pytest.raises(SyncError) as exc_info:
            pipeline.sync()
        
        assert "Git pull failed" in str(exc_info.value)
    
    @patch('sync_pipeline.SyncPipeline.qdrant_service')
    @patch('sync_pipeline.SyncPipeline.embedding_service')
    @patch('sync_pipeline.SyncPipeline.markdown_processor')
    @patch('sync_pipeline.SyncPipeline.git_watcher')
    def test_sync_embedding_error(self, mock_git, mock_md, mock_embedding, mock_qdrant):
        """Test handling of embedding generation errors."""
        from sync_pipeline import SyncPipeline
        
        # Mock changes and processing
        mock_git.has_changes.return_value = True
        mock_git.get_changed_files.return_value = (
            ['test.md'],  # added
            [],           # modified
            []            # deleted
        )
        mock_git.get_file_content.return_value = "Test content"
        mock_git.get_file_commit_info.return_value = {
            'commit_sha': 'abc123',
            'commit_date': '2024-01-01T00:00:00Z'
        }
        mock_md.process_markdown.return_value = {
            'title': 'Test',
            'content': 'Test content',
            'filename': 'test.md',
            'is_valid': True,
            'validation_error': None,
            'char_count': 100,
            'estimated_tokens': 25
        }
        mock_md.calculate_document_id.return_value = 'test123'
        mock_md.generate_metadata.return_value = {}
        
        # Mock embedding error
        mock_embedding.generate_embedding.side_effect = Exception("Embedding failed")
        mock_qdrant.collection_exists.return_value = True
        
        pipeline = SyncPipeline()
        result = pipeline.sync()
        
        assert result['status'] == 'partial_success'
        assert result['upserted'] == 0
        assert len(result['failed_files']) == 1
        assert 'test.md' in result['failed_files']
    
    @patch('sync_pipeline.SyncPipeline.qdrant_service')
    @patch('sync_pipeline.SyncPipeline.embedding_service')
    @patch('sync_pipeline.SyncPipeline.markdown_processor')
    @patch('sync_pipeline.SyncPipeline.git_watcher')
    def test_sync_qdrant_error(self, mock_git, mock_md, mock_embedding, mock_qdrant):
        """Test handling of Qdrant upsert errors."""
        from sync_pipeline import SyncPipeline
        
        # Mock changes and processing
        mock_git.has_changes.return_value = True
        mock_git.get_changed_files.return_value = (
            ['test.md'],  # added
            [],           # modified
            []            # deleted
        )
        mock_git.get_file_content.return_value = "Test content"
        mock_git.get_file_commit_info.return_value = {
            'commit_sha': 'abc123',
            'commit_date': '2024-01-01T00:00:00Z'
        }
        mock_md.process_markdown.return_value = {
            'title': 'Test',
            'content': 'Test content',
            'filename': 'test.md',
            'is_valid': True,
            'validation_error': None,
            'char_count': 100,
            'estimated_tokens': 25
        }
        mock_md.calculate_document_id.return_value = 'test123'
        mock_md.generate_metadata.return_value = {}
        mock_embedding.generate_embedding.return_value = [0.1] * 1024
        mock_qdrant.collection_exists.return_value = True
        
        # Mock Qdrant error
        mock_qdrant.upsert_point.side_effect = Exception("Qdrant upsert failed")
        
        pipeline = SyncPipeline()
        result = pipeline.sync()
        
        assert result['status'] == 'partial_success'
        assert result['upserted'] == 0
        assert len(result['failed_files']) == 1


class TestSyncPipelineReindexAll:
    """Test full reindexing functionality."""
    
    @patch('sync_pipeline.SyncPipeline.qdrant_service')
    @patch('sync_pipeline.SyncPipeline.embedding_service')
    @patch('sync_pipeline.SyncPipeline.markdown_processor')
    @patch('sync_pipeline.SyncPipeline.git_watcher')
    def test_reindex_all_success(self, mock_git, mock_md, mock_embedding, mock_qdrant):
        """Test successful full reindexing."""
        from sync_pipeline import SyncPipeline
        
        # Mock markdown files
        mock_git.get_markdown_files.return_value = ['policy1.md', 'policy2.md']
        
        def mock_get_file_content(file_path):
            return f"# Content for {file_path}"
        
        mock_git.get_file_content.side_effect = mock_get_file_content
        
        # Mock file commit info
        def mock_get_file_commit_info(file_path):
            return {
                'commit_sha': f'sha_for_{file_path}',
                'commit_date': '2024-01-01T00:00:00Z'
            }
        
        mock_git.get_file_commit_info.side_effect = mock_get_file_commit_info
        
        def mock_process_markdown(content, file_path):
            return {
                'title': f'Title for {file_path}',
                'content': content,
                'filename': file_path,
                'is_valid': True,
                'validation_error': None,
                'char_count': len(content),
                'estimated_tokens': len(content) // 4
            }
        
        def mock_calculate_document_id(file_path):
            return f'doc_{file_path.replace(".md", "")}'
        
        def mock_generate_metadata(content, title, filename, file_path, commit_info, github_url):
            return {
                'document_id': f'doc_{filename.replace(".md", "")}',
                'title': title,
                'file_path': file_path,
                'last_modified': commit_info.get('commit_date', '2024-01-01T00:00:00Z'),
                'commit_hash': commit_info.get('commit_sha', 'abc123'),
                'github_url': github_url,
                'content_length': len(content),
                'estimated_tokens': len(content) // 4,
                'content': content,
                'chunk_index': 0,
                'total_chunks': 1,
                'section_title': title,
                'chunk_tokens': len(content) // 4,
                'is_chunk': False
            }
        
        mock_md.process_markdown.side_effect = mock_process_markdown
        mock_md.calculate_document_id.side_effect = mock_calculate_document_id
        mock_md.generate_metadata.side_effect = mock_generate_metadata
        mock_embedding.generate_embedding.return_value = [0.1] * 1024
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.delete_collection.return_value = True
        mock_qdrant.create_collection.return_value = True
        mock_qdrant.upsert_point.return_value = True
        
        pipeline = SyncPipeline()
        result = pipeline.reindex_all()
        
        assert result['status'] == 'success'
        assert result['total_files'] == 2
        assert result['processed'] == 2
        assert result['failed'] == 0
        
        # Verify collection recreation
        mock_qdrant.delete_collection.assert_called_once()
        mock_qdrant.create_collection.assert_called_once()
        
        # Verify all files processed
        assert mock_md.process_markdown.call_count == 2
        assert mock_embedding.generate_embedding.call_count == 2
        assert mock_qdrant.upsert_point.call_count == 2
    
    @patch('sync_pipeline.SyncPipeline.qdrant_service')
    @patch('sync_pipeline.SyncPipeline.git_watcher')
    def test_reindex_all_no_files(self, mock_git, mock_qdrant):
        """Test reindexing when no markdown files exist."""
        from sync_pipeline import SyncPipeline
        
        # Mock no files
        mock_git.get_markdown_files.return_value = []
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.delete_collection.return_value = True
        mock_qdrant.create_collection.return_value = True
        
        pipeline = SyncPipeline()
        result = pipeline.reindex_all()
        
        assert result['status'] == 'success'
        assert result['total_files'] == 0
        assert result['processed'] == 0
        assert result['failed'] == 0


class TestSyncPipelineIntegration:
    """Integration tests for sync pipeline."""
    
    def test_sync_pipeline_imports(self):
        """Test that sync pipeline can import all required modules."""
        # Test component imports work
        from git_watcher import GitWatcher
        from markdown_processor import MarkdownProcessor
        from embedding_service import EmbeddingService
        from qdrant_service import QdrantService
        from config import Config
        
        # All imports should work without error
        assert GitWatcher is not None
        assert MarkdownProcessor is not None
        assert EmbeddingService is not None
        assert QdrantService is not None
        assert Config is not None
    
    def test_config_integration(self):
        """Test configuration integration with sync pipeline."""
        config = Config(
            repo_url="https://github.com/test/repo.git",
            branch="test",
            qdrant_collection="test_collection",
            vector_size=1024,
            embedding_model="bge-m3"
        )
        
        # Config should have all required attributes
        assert config.repo_url == "https://github.com/test/repo.git"
        assert config.branch == "test"
        assert config.qdrant_collection == "test_collection"
        assert config.vector_size == 1024
        assert config.embedding_model == "bge-m3"


# Test utilities and error classes that need to be implemented
def test_sync_error_class():
    """Test that SyncError exception class will be implemented."""
    # This test will fail until SyncError is implemented
    try:
        from sync_pipeline import SyncError
        error = SyncError("Test error", cause=Exception("Original error"))
        assert str(error) == "Test error"
        assert error.cause is not None
    except ImportError:
        # Expected to fail initially
        pytest.skip("SyncError not yet implemented")


def test_sync_pipeline_cli_interface():
    """Test CLI interface functionality."""
    # This test will fail until CLI is implemented
    try:
        from sync_pipeline import main
        # Should be able to call main without errors
        assert main is not None
    except ImportError:
        # Expected to fail initially
        pytest.skip("CLI interface not yet implemented")