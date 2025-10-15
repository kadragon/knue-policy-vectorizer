"""Tests for the standalone Cloudflare R2 sync pipeline."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config


class TestCloudflareR2SyncPipeline:
    """Validate CloudflareR2SyncPipeline behaviour."""

    @patch("r2_sync_pipeline.CloudflareR2Service")
    @patch("r2_sync_pipeline.MarkdownProcessor")
    @patch("r2_sync_pipeline.GitWatcher")
    def test_sync_upload_success(
        self,
        mock_git_cls,
        mock_md_cls,
        mock_r2_cls,
    ):
        from r2_sync_pipeline import CloudflareR2SyncPipeline

        mock_git = mock_git_cls.return_value
        mock_git.pull_updates.return_value = None
        mock_git.get_current_commit.return_value = "new"
        mock_git.has_changes.return_value = True
        mock_git.get_changed_files.return_value = (
            ["policies/new.md"],
            [],
            [],
            [],
        )
        mock_git.get_file_content.return_value = "# 제목\n\n내용"
        mock_git.get_file_commit_info.return_value = {
            "commit_sha": "abc123",
            "commit_date": "2024-01-01T00:00:00Z",
        }

        mock_md = mock_md_cls.return_value
        mock_md.process_markdown.return_value = {
            "title": "제목",
            "content": "# 제목\n\n내용",
            "filename": "policies/new.md",
            "frontmatter": {"category": "규정"},
            "is_valid": True,
            "validation_error": None,
            "char_count": 8,
            "estimated_tokens": 4,
            "needs_chunking": False,
        }
        mock_md.calculate_document_id.return_value = "doc123"

        mock_r2 = mock_r2_cls.return_value
        mock_r2.upload_document.return_value = {"key": "policies/new.md"}

        pipeline = CloudflareR2SyncPipeline(Config())
        result = pipeline.sync()

        assert result["status"] == "success"
        assert result["uploaded"] == 1
        assert result["deleted"] == 0
        mock_r2.upload_document.assert_called_once()

    @patch("r2_sync_pipeline.CloudflareR2Service")
    @patch("r2_sync_pipeline.MarkdownProcessor")
    @patch("r2_sync_pipeline.GitWatcher")
    def test_sync_handles_deletes(
        self,
        mock_git_cls,
        mock_md_cls,
        mock_r2_cls,
    ):
        from r2_sync_pipeline import CloudflareR2SyncPipeline

        mock_git = mock_git_cls.return_value
        mock_git.pull_updates.return_value = None
        mock_git.get_current_commit.return_value = "new"
        mock_git.has_changes.return_value = True
        mock_git.get_changed_files.return_value = (
            [],
            [],
            ["policies/old.md"],
            [],
        )

        mock_r2 = mock_r2_cls.return_value
        mock_r2.delete_document.return_value = {"key": "policies/old.md"}

        pipeline = CloudflareR2SyncPipeline(Config())
        result = pipeline.sync()

        assert result["status"] == "success"
        assert result["deleted"] == 1
        mock_r2.delete_document.assert_called_once_with(relative_path="policies/old.md")

    @patch("r2_sync_pipeline.CloudflareR2Service")
    @patch("r2_sync_pipeline.MarkdownProcessor")
    @patch("r2_sync_pipeline.GitWatcher")
    def test_sync_partial_failures_propagate(
        self,
        mock_git_cls,
        mock_md_cls,
        mock_r2_cls,
    ):
        from r2_sync_pipeline import CloudflareR2SyncPipeline

        mock_git = mock_git_cls.return_value
        mock_git.pull_updates.return_value = None
        mock_git.get_current_commit.return_value = "new"
        mock_git.has_changes.return_value = True
        mock_git.get_changed_files.return_value = (
            ["policies/bad.md"],
            [],
            [],
            [],
        )
        mock_git.get_file_content.return_value = "# 제목\n\n내용"
        mock_git.get_file_commit_info.return_value = {
            "commit_sha": "sha",
            "commit_date": "2024-01-01T00:00:00Z",
        }

        mock_md = mock_md_cls.return_value
        mock_md.process_markdown.return_value = {
            "title": "제목",
            "content": "# 제목\n\n내용",
            "filename": "policies/bad.md",
            "frontmatter": {},
            "is_valid": True,
            "validation_error": None,
            "char_count": 8,
            "estimated_tokens": 4,
            "needs_chunking": False,
        }
        mock_md.calculate_document_id.return_value = "doc456"

        mock_r2 = mock_r2_cls.return_value
        mock_r2.upload_document.side_effect = Exception("network error")

        pipeline = CloudflareR2SyncPipeline(Config())
        result = pipeline.sync()

        assert result["status"] == "partial_success"
        assert result["failed_files"] == ["policies/bad.md"]

    @patch("r2_sync_pipeline.GitWatcher")
    def test_sync_no_changes(self, mock_git_cls):
        from r2_sync_pipeline import CloudflareR2SyncPipeline

        mock_git = mock_git_cls.return_value
        mock_git.pull_updates.return_value = None
        mock_git.get_current_commit.return_value = "same"
        mock_git.has_changes.return_value = False

        pipeline = CloudflareR2SyncPipeline(Config())
        result = pipeline.sync()

        assert result["status"] == "success"
        assert result["changes_detected"] is False
        assert result["uploaded"] == 0
        assert result["deleted"] == 0
