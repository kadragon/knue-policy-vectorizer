"""Standalone synchronization pipeline for Cloudflare R2 uploads."""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional

import structlog

# Support both package and standalone imports
try:
    from .cloudflare_r2_service import CloudflareR2Service
    from .config import Config
    from .logger import setup_logger
    from .markdown_processor import MarkdownProcessor
except ImportError:  # pragma: no cover
    from cloudflare_r2_service import CloudflareR2Service  # type: ignore
    from config import Config  # type: ignore
    from logger import setup_logger  # type: ignore
    from markdown_processor import MarkdownProcessor  # type: ignore

logger = structlog.get_logger(__name__)


class CloudflareR2SyncError(Exception):
    """Raised when Cloudflare R2 synchronization fails."""


class CloudflareR2SyncPipeline:
    """Synchronize cleaned Markdown documents into Cloudflare R2."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        setup_logger(self.config.log_level, "CloudflareR2SyncPipeline")
        self.logger = logger.bind(pipeline="cloudflare_r2_sync")
        self._r2_file_etags: Optional[Dict[str, str]] = None

    @property
    def markdown_processor(self) -> MarkdownProcessor:
        if not hasattr(self, "_markdown_processor"):
            self._markdown_processor = MarkdownProcessor(config=self.config)
        return self._markdown_processor

    @property
    def r2_service(self) -> CloudflareR2Service:
        if not hasattr(self, "_r2_service"):
            r2_config = self.config.get_r2_service_config()
            self._r2_service = CloudflareR2Service(r2_config)
        return self._r2_service

    def _get_r2_etags(self) -> Dict[str, str]:
        """Fetch and cache all document ETags from R2."""
        if self._r2_file_etags is None:
            self.logger.debug("Fetching all document ETags from R2...")
            self._r2_file_etags = self.r2_service.list_all_documents()
            self.logger.info("ETag cache populated", count=len(self._r2_file_etags))
        return self._r2_file_etags

    @staticmethod
    def _calculate_md5(content: str) -> str:
        """Calculate the MD5 hash of the content, compatible with R2's ETag."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _build_metadata(
        self, file_path: str, processed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build metadata for R2 object from processed markdown."""
        doc_id = self.markdown_processor.calculate_document_id(file_path)
        return {
            "document_id": doc_id,
            "title": processed.get("title"),
            "source_path": file_path,
            "char_count": processed.get("char_count"),
            "frontmatter": processed.get("frontmatter", {}),
        }

    def sync(self) -> Dict[str, Any]:
        """Execute a full, ETag-based synchronization to Cloudflare R2."""
        self.logger.info("Starting Cloudflare R2 sync using ETag comparison")

        try:
            remote_etags = self._get_r2_etags()
            local_files = self._get_local_markdown_files()

            uploaded_files: List[Dict[str, Any]] = []
            skipped_files: List[str] = []
            deleted_files: List[Dict[str, Any]] = []
            failed_files: List[str] = []

            remote_keys = set(remote_etags.keys())
            processed_local_keys = set()

            # Process and upload local files
            for file_path in local_files:
                object_key = self.r2_service.build_object_key(file_path)
                processed_local_keys.add(object_key)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        raw_content = f.read()

                    processed = self.markdown_processor.process_markdown_for_r2(
                        raw_content, os.path.basename(file_path)
                    )
                    content = processed["content"]
                    local_etag = self._calculate_md5(content)

                    if remote_etags.get(object_key) == local_etag:
                        self.logger.debug("ETag match, skipping upload", file=file_path)
                        skipped_files.append(file_path)
                        continue

                    self.logger.info("Uploading modified file", file=file_path)
                    metadata = self._build_metadata(file_path, processed)
                    upload_result = self.r2_service.upload_document(
                        key=object_key, body=content, metadata=metadata
                    )
                    uploaded_files.append(
                        {
                            "file": file_path,
                            "title": processed.get("title"),
                            "r2": upload_result,
                        }
                    )

                except Exception as error:
                    self.logger.error(
                        "Failed to process or upload file",
                        file=file_path,
                        error=str(error),
                    )
                    failed_files.append(file_path)

            # Delete remote files that are no longer present locally
            files_to_delete = remote_keys - processed_local_keys
            for key_to_delete in files_to_delete:
                try:
                    self.logger.info("Deleting remote file", key=key_to_delete)
                    delete_result = self.r2_service.delete_document(key=key_to_delete)
                    deleted_files.append(
                        {"key": key_to_delete, "r2": delete_result, "status": "success"}
                    )
                except Exception as error:
                    self.logger.error(
                        "Failed to delete document from R2",
                        key=key_to_delete,
                        error=str(error),
                    )
                    failed_files.append(key_to_delete)

            status = "success" if not failed_files else "partial_success"
            self.logger.info(
                "Cloudflare R2 sync completed",
                status=status,
                uploaded=len(uploaded_files),
                skipped=len(skipped_files),
                deleted=len(deleted_files),
                failed=len(failed_files),
            )

            return {
                "status": status,
                "uploaded": len(uploaded_files),
                "skipped": len(skipped_files),
                "deleted": len(deleted_files),
                "failed": len(failed_files),
                "uploaded_files": uploaded_files,
                "deleted_files": deleted_files,
                "failed_files": failed_files,
            }

        except Exception as error:
            self.logger.error("Cloudflare R2 sync operation failed", error=str(error))
            raise CloudflareR2SyncError(str(error)) from error

    def _get_local_markdown_files(self) -> List[str]:
        """Find all markdown files in the configured repository cache directory."""
        markdown_files = []
        for root, _, files in os.walk(self.config.repo_cache_dir):
            for file in files:
                if file.endswith(".md"):
                    full_path = os.path.join(root, file)
                    markdown_files.append(full_path)
        self.logger.info("Found local markdown files", count=len(markdown_files))
        return markdown_files
