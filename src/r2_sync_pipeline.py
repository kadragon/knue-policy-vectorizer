"""Standalone synchronization pipeline for Cloudflare R2 uploads."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import structlog

try:
    from .cloudflare_r2_service import CloudflareR2Service
    from .config import Config
    from .git_watcher import GitWatcher
    from .logger import setup_logger
    from .markdown_processor import MarkdownProcessor
except Exception:  # pragma: no cover - fallback when run as script
    from cloudflare_r2_service import CloudflareR2Service  # type: ignore
    from config import Config  # type: ignore
    from git_watcher import GitWatcher  # type: ignore
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
        self._last_commit: Optional[str] = None

    @property
    def git_watcher(self) -> GitWatcher:
        if not hasattr(self, "_git_watcher"):
            git_config = {
                "repo_url": self.config.repo_url,
                "branch": self.config.branch,
                "cache_dir": self.config.repo_cache_dir,
                "log_level": self.config.log_level,
            }
            self._git_watcher = GitWatcher(git_config)
        return self._git_watcher

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

    def _build_metadata(
        self,
        file_path: str,
        processed: Dict[str, Any],
        commit_info: Dict[str, str],
        doc_id: str,
    ) -> Dict[str, Any]:
        metadata = {
            "document_id": doc_id,
            "title": processed.get("title"),
            "source_path": file_path,
            "commit_sha": commit_info.get("commit_sha"),
            "commit_date": commit_info.get("commit_date"),
            "char_count": processed.get("char_count"),
            "estimated_tokens": processed.get("estimated_tokens"),
            "needs_chunking": processed.get("needs_chunking", False),
        }
        frontmatter = processed.get("frontmatter")
        if frontmatter:
            metadata["frontmatter"] = frontmatter
        return metadata

    def _upload_document(
        self,
        file_path: str,
        processed: Dict[str, Any],
        commit_info: Dict[str, str],
        doc_id: str,
    ) -> Dict[str, Any]:
        metadata = self._build_metadata(file_path, processed, commit_info, doc_id)
        return self.r2_service.upload_document(
            relative_path=file_path,
            body=processed.get("content", ""),
            metadata=metadata,
        )

    def _delete_document(self, file_path: str) -> Dict[str, Any]:
        return self.r2_service.delete_document(relative_path=file_path)

    def sync(self) -> Dict[str, Any]:
        """Execute incremental synchronization to Cloudflare R2."""
        self.logger.info("Starting Cloudflare R2 sync")

        try:
            self.git_watcher.pull_updates()
            current_commit = self.git_watcher.get_current_commit()

            if not self.git_watcher.has_changes(self._last_commit, current_commit):
                self.logger.info("No markdown changes detected")
                return {
                    "status": "success",
                    "changes_detected": False,
                    "uploaded": 0,
                    "deleted": 0,
                    "renamed": 0,
                    "processed_files": [],
                    "deleted_files": [],
                    "renamed_files": [],
                    "failed_files": [],
                }

            added, modified, deleted, renamed = self.git_watcher.get_changed_files(
                self._last_commit, current_commit
            )

            processed_files: List[Dict[str, Any]] = []
            deleted_files: List[Dict[str, Any]] = []
            failed_files: List[str] = []

            def _handle_upload(file_path: str) -> None:
                try:
                    content = self.git_watcher.get_file_content(file_path)
                    processed = self.markdown_processor.process_markdown(
                        content, file_path
                    )

                    if not processed.get("is_valid", False) and not processed.get(
                        "needs_chunking", False
                    ):
                        raise ValueError(
                            processed.get(
                                "validation_error", "Invalid markdown content"
                            )
                        )

                    commit_info = self.git_watcher.get_file_commit_info(file_path)
                    doc_id = self.markdown_processor.calculate_document_id(file_path)
                    upload_result = self._upload_document(
                        file_path, processed, commit_info, doc_id
                    )
                    processed_files.append(
                        {
                            "file": file_path,
                            "doc_id": doc_id,
                            "title": processed.get("title"),
                            "r2": upload_result,
                        }
                    )
                except Exception as error:
                    self.logger.error(
                        "Failed to upload document to Cloudflare R2",
                        file=file_path,
                        error=str(error),
                    )
                    failed_files.append(file_path)

            def _handle_delete(file_path: str) -> None:
                try:
                    delete_result = self._delete_document(file_path)
                    deleted_files.append(
                        {"file": file_path, "r2": delete_result, "status": "success"}
                    )
                except Exception as error:
                    self.logger.error(
                        "Failed to delete document from Cloudflare R2",
                        file=file_path,
                        error=str(error),
                    )
                    failed_files.append(file_path)

            for file_path in added + modified:
                if file_path.endswith(".md"):
                    _handle_upload(file_path)

            for file_path in deleted:
                if file_path.endswith(".md"):
                    _handle_delete(file_path)

            renamed_processed: List[Tuple[str, str]] = []
            for old_path, new_path in renamed:
                if old_path and old_path.endswith(".md"):
                    _handle_delete(old_path)
                if new_path and new_path.endswith(".md"):
                    _handle_upload(new_path)
                    renamed_processed.append((old_path, new_path))

            status = "success" if not failed_files else "partial_success"
            self._last_commit = current_commit

            self.logger.info(
                "Cloudflare R2 sync completed",
                status=status,
                uploaded=len(processed_files),
                deleted=len(deleted_files),
                renamed=len(renamed_processed),
                failed=len(failed_files),
            )

            return {
                "status": status,
                "changes_detected": True,
                "uploaded": len(processed_files),
                "deleted": len(deleted_files),
                "renamed": len(renamed_processed),
                "processed_files": processed_files,
                "deleted_files": deleted_files,
                "renamed_files": renamed_processed,
                "failed_files": failed_files,
            }

        except Exception as error:
            self.logger.error("Cloudflare R2 sync operation failed", error=str(error))
            raise CloudflareR2SyncError(str(error)) from error
