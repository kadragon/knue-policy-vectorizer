"""Tests for Cloudflare R2 service integration."""

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestCloudflareR2Service:
    """Validate CloudflareR2Service behavior."""

    def _service(self, **overrides):
        from cloudflare_r2_service import CloudflareR2Service

        config = {
            "bucket": "knue-vectorstore",
            "endpoint": "https://example.r2.cloudflarestorage.com/knue-vectorstore",
            "access_key_id": "key",
            "secret_access_key": "secret",
            "key_prefix": "policies",
            "soft_delete_enabled": overrides.get("soft_delete_enabled", False),
            "soft_delete_prefix": overrides.get("soft_delete_prefix", "deleted/"),
        }
        config.update(overrides)
        mock_client = MagicMock()
        service = CloudflareR2Service(config, s3_client=mock_client)
        return service, mock_client

    def test_upload_document_serializes_metadata(self):
        from cloudflare_r2_service import DEFAULT_CONTENT_TYPE

        service, mock_client = self._service()

        result = service.upload_document(
            relative_path="규정/학칙.md",
            body="# 내용",
            metadata={"frontmatter": {"title": "학칙", "tags": ["규정"]}},
        )

        mock_client.put_object.assert_called_once()
        call_kwargs = mock_client.put_object.call_args.kwargs

        assert call_kwargs["Bucket"] == "knue-vectorstore"
        assert call_kwargs["Key"] == "policies/규정/학칙.md"
        assert call_kwargs["ContentType"] == DEFAULT_CONTENT_TYPE
        assert call_kwargs["Body"] == "# 내용".encode("utf-8")
        assert json.loads(call_kwargs["Metadata"]["frontmatter"])["title"] == "학칙"
        assert result["key"] == "policies/규정/학칙.md"

    def test_delete_document_soft_delete(self):
        service, mock_client = self._service(
            soft_delete_enabled=True, soft_delete_prefix="archive"
        )

        service.delete_document(relative_path="rules/rule.md")

        assert mock_client.copy_object.called
        copy_kwargs = mock_client.copy_object.call_args.kwargs
        assert copy_kwargs["Bucket"] == "knue-vectorstore"
        assert copy_kwargs["CopySource"]["Key"] == "policies/rules/rule.md"
        archive_key = copy_kwargs["Key"]
        assert archive_key.startswith("archive/")

        mock_client.delete_object.assert_called_once()
        delete_kwargs = mock_client.delete_object.call_args.kwargs
        assert delete_kwargs["Key"] == "policies/rules/rule.md"

    def test_build_object_key_handles_prefix(self):
        service, _ = self._service(key_prefix="docs")
        key = service.build_object_key("./가이드\\문서.md")
        assert key == "docs/가이드/문서.md"

    def test_upload_document_retries_on_transient_failure(self):
        service, mock_client = self._service()
        mock_client.put_object.side_effect = [
            Exception("network glitch"),
            Exception("another glitch"),
            {"VersionId": "v3"},
        ]

        result = service.upload_document(relative_path="규정/학칙.md", body="본문")

        assert mock_client.put_object.call_count == 3
        assert result["version_id"] == "v3"

    def test_upload_document_raises_after_retry_exhaustion(self):
        service, mock_client = self._service()
        mock_client.put_object.side_effect = Exception("fatal error")

        with pytest.raises(Exception):
            service.upload_document(relative_path="규정/학칙.md", body="본문")

        assert mock_client.put_object.call_count == 3

    def test_missing_bucket_raises(self):
        from cloudflare_r2_service import CloudflareR2Service

        with pytest.raises(ValueError):
            CloudflareR2Service(
                {
                    "bucket": "",
                    "endpoint": "https://example.r2.cloudflarestorage.com",
                    "access_key_id": "key",
                    "secret_access_key": "secret",
                }
            )
