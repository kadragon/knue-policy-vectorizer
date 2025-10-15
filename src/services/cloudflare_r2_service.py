"""Cloudflare R2 storage integration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

import boto3
import structlog
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

logger = structlog.get_logger(__name__)

DEFAULT_CONTENT_TYPE = "text/markdown; charset=utf-8"


class CloudflareR2Service:
    """Lightweight wrapper around the Cloudflare R2 S3-compatible API."""

    def __init__(
        self,
        config: Dict[str, Any],
        s3_client: Optional[Any] = None,
    ) -> None:
        """Initialize the R2 service with configuration.

        Args:
            config: Dict with required keys (bucket, access credentials, endpoint).
            s3_client: Optional pre-configured S3 client (useful for testing).
        """
        self.logger = logger.bind(component="CloudflareR2Service")

        self.account_id: str = config.get("account_id", "")

        self.bucket: str = config.get("bucket", "")
        if not self.bucket:
            raise ValueError("Cloudflare R2 bucket must be configured")

        self.key_prefix: str = (config.get("key_prefix") or "").strip().strip("/")
        self.soft_delete_enabled: bool = bool(config.get("soft_delete_enabled", False))
        self.soft_delete_prefix: str = (
            config.get("soft_delete_prefix") or "deleted/"
        ).strip() or "deleted"

        endpoint = config.get("endpoint") or ""
        self.endpoint_url = self._normalize_endpoint(endpoint)

        self.logger = self.logger.bind(bucket=self.bucket, endpoint=self.endpoint_url)

        if s3_client is not None:
            self._s3 = s3_client
            return

        access_key_id = config.get("access_key_id")
        secret_access_key = config.get("secret_access_key")
        if not access_key_id or not secret_access_key:
            raise ValueError(
                "Cloudflare R2 access key id and secret access key must be configured"
            )

        session = boto3.session.Session()
        boto_config = BotoConfig(signature_version="s3v4")

        # Cloudflare R2 expects region "auto"
        self._s3 = session.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name="auto",
            config=boto_config,
        )

    def _normalize_endpoint(self, endpoint: str) -> str:
        """Ensure endpoint does not contain bucket suffixes and ends without a slash."""
        endpoint = (endpoint or "").strip()
        if not endpoint:
            if not self.account_id:
                raise ValueError(
                    "Cloudflare R2 endpoint not provided and account_id unknown"
                )
            endpoint = f"https://{self.account_id}.r2.cloudflarestorage.com"

        endpoint = endpoint.rstrip("/")
        bucket_suffix = f"/{self.bucket}"
        if endpoint.endswith(bucket_suffix):
            endpoint = endpoint[: -len(bucket_suffix)]
        return endpoint

    def _build_object_key(self, relative_path: str) -> str:
        """Convert repository-relative path to object key with optional prefix."""
        normalized = relative_path.strip().lstrip("./")
        normalized = normalized.replace("\\", "/")
        if self.key_prefix:
            return f"{self.key_prefix}/{normalized}"
        return normalized

    def _build_soft_delete_key(self, object_key: str) -> str:
        """Construct soft-delete archive key."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        prefix = self.soft_delete_prefix.rstrip("/") + "/"
        return f"{prefix}{object_key}.{timestamp}"

    @staticmethod
    def _prepare_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Coerce metadata values to strings and flatten nested structures."""
        if not metadata:
            return {}

        prepared: Dict[str, str] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (dict, list, tuple)):
                prepared_value = json.dumps(value, ensure_ascii=True, default=str)
            else:
                prepared_value = str(value)
                if not prepared_value.isascii():
                    prepared_value = prepared_value.encode("unicode_escape").decode(
                        "ascii"
                    )
            prepared[key] = prepared_value
        return prepared

    def _execute_with_retry(
        self,
        operation: Callable[[], Any],
        *,
        action: str,
        key: str,
        max_attempts: int = 3,
    ) -> Any:
        """Execute an S3 operation with simple retry logic."""
        attempt = 1
        while True:
            try:
                return operation()
            except Exception as error:
                if attempt >= max_attempts:
                    self.logger.error(
                        "R2 operation failed after retries",
                        action=action,
                        key=key,
                        attempts=attempt,
                        error=str(error),
                    )
                    raise

                self.logger.warning(
                    "R2 operation failed, retrying",
                    action=action,
                    key=key,
                    attempt=attempt,
                    error=str(error),
                )
                attempt += 1

    def upload_document(
        self,
        *,
        relative_path: Optional[str] = None,
        key: Optional[str] = None,
        body: str,
        metadata: Optional[Dict[str, Any]] = None,
        content_type: str = DEFAULT_CONTENT_TYPE,
    ) -> Dict[str, Any]:
        """Upload a markdown document to R2."""
        if key is None:
            if relative_path is None:
                raise ValueError("Either relative_path or key must be provided")
            object_key = self._build_object_key(relative_path)
        else:
            object_key = key

        metadata_payload = self._prepare_metadata(metadata)

        self.logger.debug(
            "Uploading object to R2",
            key=object_key,
            metadata_keys=list(metadata_payload.keys()),
        )

        response = self._execute_with_retry(
            lambda: self._s3.put_object(
                Bucket=self.bucket,
                Key=object_key,
                Body=body.encode("utf-8"),
                ContentType=content_type,
                Metadata=metadata_payload,
            ),
            action="upload",
            key=object_key,
        )
        return {"key": object_key, "version_id": response.get("VersionId")}

    def delete_document(
        self,
        *,
        relative_path: Optional[str] = None,
        key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete (or archive) a document in R2."""
        if key is None:
            if relative_path is None:
                raise ValueError("Either relative_path or key must be provided")
            object_key = self._build_object_key(relative_path)
        else:
            object_key = key

        archive_key = None
        if self.soft_delete_enabled:
            archive_key = self._build_soft_delete_key(object_key)
            try:
                self.logger.debug(
                    "Archiving object prior to delete",
                    source_key=object_key,
                    archive_key=archive_key,
                )
                self._execute_with_retry(
                    lambda: self._s3.copy_object(
                        Bucket=self.bucket,
                        CopySource={"Bucket": self.bucket, "Key": object_key},
                        Key=archive_key,
                    ),
                    action="soft_delete_copy",
                    key=object_key,
                )
            except ClientError as error:
                error_code = error.response.get("Error", {}).get("Code")
                if error_code == "NoSuchKey":
                    self.logger.warning(
                        "Object missing during soft-delete copy",
                        key=object_key,
                        error=str(error),
                    )
                else:
                    raise

        self.logger.debug("Deleting object from R2", key=object_key)
        response = self._execute_with_retry(
            lambda: self._s3.delete_object(Bucket=self.bucket, Key=object_key),
            action="delete",
            key=object_key,
        )
        return {
            "key": object_key,
            "archive_key": archive_key,
            "version_id": response.get("VersionId"),
        }

    def list_all_documents(self) -> Dict[str, str]:
        """List all documents in the R2 bucket, returning a map of key to ETag."""
        self.logger.debug("Listing all documents in R2 bucket")
        all_objects: Dict[str, str] = {}
        continuation_token = None

        while True:
            try:
                args = {"Bucket": self.bucket}
                if self.key_prefix:
                    args["Prefix"] = self.key_prefix
                if continuation_token:
                    args["ContinuationToken"] = continuation_token

                response = self._s3.list_objects_v2(**args)
                for obj in response.get("Contents", []):
                    key = obj.get("Key")
                    etag = (obj.get("ETag") or "").strip('"')
                    if key:
                        all_objects[key] = etag

                if not response.get("IsTruncated"):
                    break
                continuation_token = response.get("NextContinuationToken")

            except ClientError as error:
                self.logger.error("Failed to list objects from R2", error=str(error))
                raise

        self.logger.info("Finished listing documents", count=len(all_objects))
        return all_objects

    def get_document_head(self, *, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a document in R2."""
        try:
            response = self._execute_with_retry(
                lambda: self._s3.head_object(Bucket=self.bucket, Key=key),
                action="head",
                key=key,
            )
            # ETag from head_object is double-quoted, remove them.
            if "ETag" in response:
                response["ETag"] = response["ETag"].strip('"')
            return response
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return None  # Object not found
            self.logger.error(
                "Failed to get document metadata from R2", key=key, error=str(e)
            )
            raise

    def build_object_key(self, relative_path: str) -> str:
        """Public helper for building object keys."""
        return self._build_object_key(relative_path)
