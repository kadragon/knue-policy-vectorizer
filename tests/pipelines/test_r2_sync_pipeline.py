"""Tests for the standalone Cloudflare R2 sync pipeline."""

import hashlib
import os
import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.config.config import Config
from src.pipelines.r2_sync_pipeline import CloudflareR2SyncPipeline


@pytest.fixture
def mock_config():
    """Provides a mock Config object."""
    config = Config()
    config.repo_cache_dir = "/fake/repo"
    return config


def calculate_md5(content):
    return hashlib.md5(content.encode("utf-8")).hexdigest()


class TestCloudflareR2SyncPipeline:
    """Validate CloudflareR2SyncPipeline behaviour with ETag-based logic."""

    @patch("src.pipelines.r2_sync_pipeline.CloudflareR2Service")
    @patch("src.pipelines.r2_sync_pipeline.MarkdownProcessor")
    @patch("os.walk")
    @patch("builtins.open")
    def test_sync_new_and_modified_files(
        self, mock_open_file, mock_os_walk, mock_md_cls, mock_r2_cls, mock_config
    ):
        # Setup: one new file, one modified file, one unchanged file
        mock_os_walk.return_value = [
            ("/fake/repo", [], ["new.md", "modified.md", "same.md"])
        ]

        # Content for each file
        file_contents = {
            "/fake/repo/new.md": "# New File",
            "/fake/repo/modified.md": "# Modified Content",
            "/fake/repo/same.md": "# Unchanged Content",
        }
        mock_open_file.side_effect = lambda path, *args, **kwargs: mock_open(
            read_data=file_contents[path]
        ).return_value

        # Mock R2 ETags
        mock_r2 = mock_r2_cls.return_value
        mock_r2.list_all_documents.return_value = {
            "prefix/modified.md": calculate_md5("# Old Content"),
            "prefix/same.md": calculate_md5("# Unchanged Content"),
        }
        mock_r2.build_object_key.side_effect = lambda x: f"prefix/{os.path.basename(x)}"

        # Mock MarkdownProcessor
        mock_md = mock_md_cls.return_value
        mock_md.process_markdown_for_r2.side_effect = lambda content, fname: {
            "content": content,
            "title": fname,
            "frontmatter": {},
            "char_count": len(content),
        }

        # Execute
        pipeline = CloudflareR2SyncPipeline(mock_config)
        result = pipeline.sync()

        # Assert
        assert result["status"] == "success"
        assert result["uploaded"] == 2
        assert result["skipped"] == 1
        assert result["deleted"] == 0
        assert mock_r2.upload_document.call_count == 2

    @patch("src.pipelines.r2_sync_pipeline.CloudflareR2Service")
    @patch("src.pipelines.r2_sync_pipeline.MarkdownProcessor")
    @patch("os.walk")
    @patch("builtins.open")
    def test_sync_deletes_stale_files(
        self, mock_open_file, mock_os_walk, mock_md_cls, mock_r2_cls, mock_config
    ):
        # Setup: local has one file, remote has two
        mock_os_walk.return_value = [("/fake/repo", [], ["existing.md"])]
        file_contents = {"/fake/repo/existing.md": "# Existing"}
        mock_open_file.side_effect = lambda path, *args, **kwargs: mock_open(
            read_data=file_contents[path]
        ).return_value

        # Mock R2 ETags for two files
        mock_r2 = mock_r2_cls.return_value
        remote_etags = {
            "prefix/existing.md": "etag1",
            "prefix/stale.md": "etag2",
        }
        mock_r2.list_all_documents.return_value = remote_etags
        mock_r2.build_object_key.side_effect = lambda x: f"prefix/{os.path.basename(x)}"

        # Mock MarkdownProcessor
        mock_md = mock_md_cls.return_value
        mock_md.process_markdown_for_r2.return_value = {
            "content": "# Existing",
            "title": "existing.md",
            "frontmatter": {},
            "char_count": 10,
        }

        # Execute
        pipeline = CloudflareR2SyncPipeline(mock_config)
        result = pipeline.sync()

        # Assert
        assert result["status"] == "success"
        assert result["deleted"] == 1
        assert result["deleted_files"][0]["key"] == "prefix/stale.md"
        mock_r2.delete_document.assert_called_once_with(key="prefix/stale.md")

    @patch("src.pipelines.r2_sync_pipeline.CloudflareR2Service")
    @patch("src.pipelines.r2_sync_pipeline.MarkdownProcessor")
    @patch("os.walk")
    @patch("builtins.open")
    def test_sync_no_changes(
        self, mock_open_file, mock_os_walk, mock_md_cls, mock_r2_cls, mock_config
    ):
        # Setup: local and remote are identical
        mock_os_walk.return_value = [("/fake/repo", [], ["file.md"])]

        content = "# Content"
        etag = calculate_md5(content)
        file_contents = {"/fake/repo/file.md": content}
        mock_open_file.side_effect = lambda path, *args, **kwargs: mock_open(
            read_data=file_contents[path]
        ).return_value

        mock_r2 = mock_r2_cls.return_value
        mock_r2.list_all_documents.return_value = {"prefix/file.md": etag}
        mock_r2.build_object_key.return_value = "prefix/file.md"

        mock_md = mock_md_cls.return_value
        mock_md.process_markdown_for_r2.return_value = {
            "content": content,
            "title": "file.md",
            "frontmatter": {},
            "char_count": len(content),
        }

        # Execute
        pipeline = CloudflareR2SyncPipeline(mock_config)
        result = pipeline.sync()

        # Assert
        assert result["status"] == "success"
        assert result["uploaded"] == 0
        assert result["skipped"] == 1
        assert result["deleted"] == 0
        mock_r2.upload_document.assert_not_called()
        mock_r2.delete_document.assert_not_called()

    @patch("src.pipelines.r2_sync_pipeline.CloudflareR2Service")
    @patch("src.pipelines.r2_sync_pipeline.MarkdownProcessor")
    @patch("os.walk")
    @patch("builtins.open")
    def test_sync_partial_failure(
        self, mock_open_file, mock_os_walk, mock_md_cls, mock_r2_cls, mock_config
    ):
        # Setup: one file to upload, but R2 fails
        mock_os_walk.return_value = [("/fake/repo", [], ["new.md"])]
        file_contents = {"/fake/repo/new.md": "# New Content"}
        mock_open_file.side_effect = lambda path, *args, **kwargs: mock_open(
            read_data=file_contents[path]
        ).return_value

        mock_r2 = mock_r2_cls.return_value
        mock_r2.list_all_documents.return_value = {}
        mock_r2.build_object_key.return_value = "prefix/new.md"
        mock_r2.upload_document.side_effect = Exception("R2 is down")

        mock_md = mock_md_cls.return_value
        mock_md.process_markdown_for_r2.return_value = {
            "content": "# New Content",
            "title": "new.md",
            "frontmatter": {},
            "char_count": 13,
        }

        # Execute
        pipeline = CloudflareR2SyncPipeline(mock_config)
        result = pipeline.sync()

        # Assert
        assert result["status"] == "partial_success"
        assert result["uploaded"] == 0
        assert result["failed"] == 1
        assert result["failed_files"] == ["new.md"]

    @patch("src.pipelines.r2_sync_pipeline.CloudflareR2Service")
    @patch("src.pipelines.r2_sync_pipeline.MarkdownProcessor")
    @patch("os.walk")
    def test_get_local_markdown_files_excludes_readme(
        self, mock_os_walk, mock_md_cls, mock_r2_cls, mock_config
    ):
        # Setup: directory with markdown files including README
        mock_os_walk.return_value = [
            ("/fake/repo", [], ["policy.md", "README.md", "readme.md", "guide.md"])
        ]

        # Execute
        pipeline = CloudflareR2SyncPipeline(mock_config)
        files = pipeline._get_local_markdown_files()

        # Assert: README files are excluded, others included
        assert set(files) == {"policy.md", "guide.md"}
