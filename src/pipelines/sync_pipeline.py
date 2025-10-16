"""Main synchronization pipeline for KNUE Policy Hub to Qdrant."""

import hashlib
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import structlog

# Support both package and standalone imports
try:
    from src.config.config import Config
    from src.config.config_manager import (
        ConfigProfile,
        ConfigTemplate,
        ConfigurationManager,
    )
    from src.core.git_watcher import GitWatcher
    from src.core.migration_tools import MigrationManager, create_migration_config
    from src.pipelines.r2_sync_pipeline import (
        CloudflareR2SyncError,
        CloudflareR2SyncPipeline,
    )
    from src.services.embedding_service_openai import OpenAIEmbeddingService
    from src.services.knue_board_ingestor import KnueBoardIngestor
    from src.services.qdrant_service import QdrantService
    from src.utils.logger import setup_logger
    from src.utils.markdown_processor import MarkdownProcessor
    from src.utils.providers import (
        EmbeddingProvider,
        EmbeddingServiceInterface,
        ProviderFactory,
        VectorProvider,
        VectorServiceInterface,
        get_available_embedding_providers,
        get_available_vector_providers,
    )
except ImportError:  # pragma: no cover - fallback when executed as script
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from src.config.config import Config
    from src.config.config_manager import (
        ConfigProfile,
        ConfigTemplate,
        ConfigurationManager,
    )
    from src.core.git_watcher import GitWatcher
    from src.core.migration_tools import (
        MigrationManager,
        create_migration_config,
    )
    from src.pipelines.r2_sync_pipeline import (
        CloudflareR2SyncError,
        CloudflareR2SyncPipeline,
    )
    from src.services.embedding_service_openai import (
        OpenAIEmbeddingService,
    )
    from src.services.knue_board_ingestor import KnueBoardIngestor
    from src.services.qdrant_service import QdrantService
    from src.utils.logger import setup_logger
    from src.utils.markdown_processor import MarkdownProcessor
    from src.utils.providers import (
        EmbeddingProvider,
        EmbeddingServiceInterface,
        ProviderFactory,
        VectorProvider,
        VectorServiceInterface,
        get_available_embedding_providers,
        get_available_vector_providers,
    )

logger = structlog.get_logger(__name__)


class SyncError(Exception):
    """Custom exception for sync pipeline errors."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause


class SyncPipeline:
    """Main synchronization pipeline orchestrating all components."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize the sync pipeline with configuration."""
        self.config = config or Config()
        # Configure logging using configured level
        setup_logger(self.config.log_level, "SyncPipeline")
        self.logger = logger.bind(pipeline="sync")

        # Simple state tracking for commits
        self._last_commit: Optional[str] = None

    @property
    def git_watcher(self) -> GitWatcher:
        """Get or create GitWatcher instance."""
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
        """Get or create MarkdownProcessor instance."""
        if not hasattr(self, "_markdown_processor"):
            self._markdown_processor = MarkdownProcessor(config=self.config)
        return self._markdown_processor

    @property
    def embedding_service(self) -> EmbeddingServiceInterface:
        """Get or create embedding service instance using provider factory."""
        if not hasattr(self, "_embedding_service"):
            factory = ProviderFactory()
            embedding_config = self.config.get_embedding_service_config()
            self._embedding_service = factory.get_embedding_service(
                self.config.embedding_provider, embedding_config
            )
        return self._embedding_service

    @property
    def qdrant_service(self) -> VectorServiceInterface:
        """Get or create vector service instance using provider factory."""
        if not hasattr(self, "_qdrant_service"):
            factory = ProviderFactory()
            vector_config = self.config.get_vector_service_config()
            # Add common configuration required by both local and cloud services
            vector_config.update(
                {
                    "collection_name": self.config.qdrant_collection,
                    "vector_size": self.config.vector_size,
                }
            )
            self._qdrant_service = factory.get_vector_service(
                self.config.vector_provider, vector_config
            )
        return self._qdrant_service

    def health_check(self) -> bool:
        """Check if all services are healthy."""
        self.logger.info("Performing health check")

        # Check embedding service
        if not self.embedding_service.health_check():
            self.logger.error("Embedding service health check failed")
            return False

        # Check Qdrant service
        if not self.qdrant_service.health_check():
            self.logger.error("Qdrant service health check failed")
            return False

        self.logger.info("All services are healthy")
        return True

    def _ensure_collection_exists(self) -> bool:
        """Ensure the Qdrant collection exists."""
        if not self.qdrant_service.collection_exists(self.config.qdrant_collection):
            self.logger.info(
                "Creating collection", collection=self.config.qdrant_collection
            )
            try:
                self.qdrant_service.create_collection(
                    self.config.qdrant_collection, self.config.vector_size
                )
                return True
            except Exception as e:
                self.logger.error("Failed to create collection", error=str(e))
                return False

        return True

    def _process_file(self, file_path: str, action: str) -> Dict[str, Any]:
        """Process a single file for sync."""
        self.logger.debug("Processing file", file=file_path, action=action)

        try:
            if action == "delete":
                # For deleted files, delete all chunks
                doc_id = self.markdown_processor.calculate_document_id(file_path)
                self.qdrant_service.delete_document_chunks(doc_id)  # type: ignore[attr-defined]
                return {
                    "file": file_path,
                    "action": "delete",
                    "doc_id": doc_id,
                    "status": "success",
                }

            else:
                # For add/modify, process the full pipeline
                content = self.git_watcher.get_file_content(file_path)

                # Process markdown
                processed = self.markdown_processor.process_markdown(content, file_path)

                # Check if content is valid
                if not processed["is_valid"] and not processed.get(
                    "needs_chunking", False
                ):
                    raise Exception(processed["validation_error"])

                # Get commit info and GitHub URL
                file_commit_info = self.git_watcher.get_file_commit_info(file_path)
                commit_info = {"sha": file_commit_info["commit_sha"]}
                github_url = f"{self.config.repo_url.replace('.git', '')}/blob/{self.config.branch}/{file_path}"

                # Calculate document ID (used for both chunked and single documents)
                doc_id = self.markdown_processor.calculate_document_id(file_path)

                # Handle chunked content
                if processed.get("needs_chunking", False):
                    chunks = processed["chunks"]
                    self._process_chunks(
                        chunks, processed, file_path, commit_info, github_url
                    )
                else:
                    # Single document processing
                    metadata = self.markdown_processor.generate_metadata(
                        processed["content"],
                        processed["title"],
                        processed["filename"],
                        file_path,
                        commit_info,
                        github_url,
                    )

                    # Add non-chunk metadata
                    metadata.update(
                        {
                            "chunk_index": 0,
                            "total_chunks": 1,
                            "section_title": "",
                            "chunk_tokens": processed["estimated_tokens"],
                            "is_chunk": False,
                        }
                    )

                    # Generate embedding with fallback to chunking if token limit exceeded
                    try:
                        embedding = self.embedding_service.generate_embedding(
                            processed["content"]
                        )

                        # Upsert to Qdrant
                        self.qdrant_service.upsert_points(
                            self.config.qdrant_collection,
                            [
                                {
                                    "id": doc_id,
                                    "vector": embedding,
                                    "payload": metadata,
                                }
                            ],
                        )
                    except ValueError as e:
                        if "exceeds maximum token limit" in str(e):
                            self.logger.warning(
                                "Single document exceeded token limit, forcing chunking",
                                file_path=file_path,
                                error=str(e),
                            )
                            # Force chunk the content and process as chunks
                            chunks = self.markdown_processor.chunk_markdown_content(
                                processed["content"]
                            )
                            self._process_chunks(
                                chunks, processed, file_path, commit_info, github_url
                            )
                        else:
                            # Re-raise if it's a different error
                            raise

                return {
                    "file": file_path,
                    "action": action,
                    "doc_id": doc_id,
                    "title": processed["title"],
                    "status": "success",
                }

        except Exception as e:
            self.logger.error("Failed to process file", file=file_path, error=str(e))
            return {
                "file": file_path,
                "action": action,
                "status": "failed",
                "error": str(e),
            }

    def _process_chunks(
        self,
        chunks: List[Dict],
        processed: Dict,
        file_path: str,
        commit_info: Dict[str, str],
        github_url: str,
    ) -> None:
        """
        Process a list of chunks and upsert them to Qdrant using batch operations.

        Args:
            chunks: List of chunk dictionaries with content, chunk_index, section_title, tokens
            processed: Processed markdown data containing title, filename
            file_path: Path to the file being processed
            commit_info: Git commit information
            github_url: GitHub URL for the file
        """
        self.logger.info(
            "Processing chunked document", file_path=file_path, chunk_count=len(chunks)
        )

        # Prepare data for batch processing
        chunk_texts = []
        points_data = []
        base_id = self.markdown_processor.calculate_document_id(file_path)

        # First pass: prepare all chunk data and collect texts for batch embedding
        for chunk in chunks:
            chunk_texts.append(chunk["content"])

            # Generate unique UUID for chunk based on file path and chunk index
            chunk_data = f"{base_id}_chunk_{chunk['chunk_index']}"
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_data))

            # Generate metadata for chunk
            metadata = self.markdown_processor.generate_metadata(
                chunk["content"],
                processed["title"],
                processed["filename"],
                file_path,
                commit_info,
                github_url,
            )

            # Add chunk-specific metadata
            metadata.update(
                {
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": len(chunks),
                    "section_title": chunk["section_title"],
                    "chunk_tokens": chunk["tokens"],
                    "is_chunk": True,
                }
            )

            # Store prepared point data (will add embedding vector later)
            points_data.append({"point_id": chunk_id, "metadata": metadata})

        # Generate embeddings for all chunks in batch
        self.logger.debug("Generating batch embeddings", chunk_count=len(chunk_texts))
        embeddings = self.embedding_service.generate_embeddings_batch(chunk_texts)

        # Second pass: combine embeddings with point data
        batch_points = [
            {
                "point_id": point_data["point_id"],
                "vector": embedding,
                "metadata": point_data["metadata"],
            }
            for point_data, embedding in zip(points_data, embeddings)
        ]

        # Batch upsert all chunks to Qdrant
        self.logger.debug("Batch upserting chunks", chunk_count=len(batch_points))
        self.qdrant_service.upsert_points(self.config.qdrant_collection, batch_points)

    def sync(self) -> Dict[str, Any]:
        """Perform incremental synchronization."""
        self.logger.info("Starting sync operation")

        try:
            # Pull latest changes
            self.git_watcher.pull_updates()

            # Get current commit
            current_commit = self.git_watcher.get_current_commit()

            # Check for changes
            if not self.git_watcher.has_changes(self._last_commit, current_commit):
                self.logger.info("No changes detected")
                return {
                    "status": "success",
                    "changes_detected": False,
                    "upserted": 0,
                    "deleted": 0,
                    "renamed": 0,
                    "processed_files": [],
                    "deleted_files": [],
                    "renamed_files": [],
                    "failed_files": [],
                }

            # Ensure collection exists
            if not self._ensure_collection_exists():
                raise SyncError("Failed to ensure collection exists")

            # Get changed files
            added_files, modified_files, deleted_files_list, renamed_files = (
                self.git_watcher.get_changed_files(self._last_commit, current_commit)
            )

            processed_files = []
            deleted_files = []
            failed_files = []

            # Process added and modified files
            for file_path in added_files + modified_files:
                if file_path.endswith(".md"):
                    result = self._process_file(file_path, "upsert")
                    if result["status"] == "success":
                        processed_files.append(result)
                    else:
                        failed_files.append(file_path)

            # Process deleted files
            for file_path in deleted_files_list:
                if file_path.endswith(".md"):
                    result = self._process_file(file_path, "delete")
                    if result["status"] == "success":
                        deleted_files.append(result)
                    else:
                        failed_files.append(file_path)

            # Process renamed files (delete old + add new)
            for old_path, new_path in renamed_files:
                # Delete old path if it was a markdown file
                if old_path and old_path.endswith(".md"):
                    result = self._process_file(old_path, "delete")
                    if result["status"] == "success":
                        deleted_files.append(result)
                        self.logger.info(
                            "Deleted old path from rename",
                            old_path=old_path,
                            new_path=new_path,
                        )
                    else:
                        failed_files.append(old_path)
                        self.logger.error(
                            "Failed to delete old path from rename",
                            old_path=old_path,
                            new_path=new_path,
                        )

                # Add new path if it's a markdown file
                if new_path and new_path.endswith(".md"):
                    result = self._process_file(new_path, "upsert")
                    if result["status"] == "success":
                        processed_files.append(result)
                        self.logger.info(
                            "Added new path from rename",
                            old_path=old_path,
                            new_path=new_path,
                        )
                    else:
                        failed_files.append(new_path)
                        self.logger.error(
                            "Failed to add new path from rename",
                            old_path=old_path,
                            new_path=new_path,
                        )

            # Determine final status
            if failed_files:
                status = "partial_success"
            else:
                status = "success"

            result = {
                "status": status,
                "changes_detected": True,
                "upserted": len(processed_files),
                "deleted": len(deleted_files),
                "renamed": len(renamed_files),
                "processed_files": processed_files,
                "deleted_files": deleted_files,
                "renamed_files": renamed_files,
                "failed_files": failed_files,
            }

            # Update last commit
            self._last_commit = current_commit

            self.logger.info(
                "Sync completed",
                status=status,
                upserted=len(processed_files),
                deleted=len(deleted_files),
                renamed=len(renamed_files),
                failed=len(failed_files),
            )

            return result

        except Exception as e:
            self.logger.error("Sync operation failed", error=str(e))
            raise SyncError(f"Sync operation failed: {str(e)}", cause=e)

    def reindex_all(self) -> Dict[str, Any]:
        """Perform full reindexing of all markdown files."""
        self.logger.info("Starting full reindex operation")

        try:
            # Get all markdown files
            files = self.git_watcher.get_markdown_files()

            if not files:
                self.logger.info("No markdown files found")
                return {
                    "status": "success",
                    "total_files": 0,
                    "processed": 0,
                    "failed": 0,
                    "processed_files": [],
                    "failed_files": [],
                }

            # Recreate collection (delete + create)
            collection_name = self.config.qdrant_collection
            if self.qdrant_service.collection_exists(collection_name):
                self.logger.info(
                    "Deleting existing collection", collection=collection_name
                )
                self.qdrant_service.delete_collection(collection_name)

            self.logger.info("Creating new collection", collection=collection_name)
            try:
                self.qdrant_service.create_collection(
                    collection_name, self.config.vector_size
                )
            except Exception as e:
                raise SyncError("Failed to create collection") from e

            # Process all files
            processed_files = []
            failed_files = []

            for file_path in files:
                result = self._process_file(file_path, "upsert")
                if result["status"] == "success":
                    processed_files.append(result)
                else:
                    failed_files.append(file_path)

            result = {
                "status": "success",
                "total_files": len(files),
                "processed": len(processed_files),
                "failed": len(failed_files),
                "processed_files": processed_files,
                "failed_files": failed_files,
            }

            self.logger.info(
                "Reindex completed",
                total=len(files),
                processed=len(processed_files),
                failed=len(failed_files),
            )

            return result

        except Exception as e:
            self.logger.error("Reindex operation failed", error=str(e))
            raise SyncError(f"Reindex operation failed: {str(e)}", cause=e)


@click.group()
def main() -> None:
    """
    KNUE Policy Hub to Qdrant synchronization tool.

    🔒 SECURITY NOTICE:
    This tool handles API keys and sensitive credentials. By default, credentials
    are masked in exports for security. Use --include-secrets flags with caution.
    """
    pass


@main.command(name="list-providers")
def list_providers() -> None:
    """List all available embedding and vector providers."""
    click.echo("🔧 Available Providers\n")

    click.echo("📊 Available Embedding Providers:")
    for provider in get_available_embedding_providers():
        click.echo(f"  • {provider}")

    click.echo("\n🗄️ Available Vector Providers:")
    for provider in get_available_vector_providers():
        click.echo(f"  • {provider}")


@main.command(name="configure")
def configure_providers() -> None:
    """Interactive configuration of embedding and vector providers."""
    click.echo("🔧 Multi-Provider Configuration\n")

    # Get current config as defaults
    current_config = Config.from_env()

    # Embedding provider selection
    click.echo("📊 Select Embedding Provider:")
    for i, provider in enumerate(get_available_embedding_providers(), 1):
        default_marker = (
            " (current)" if provider == str(current_config.embedding_provider) else ""
        )
        click.echo(f"  {i}. {provider}{default_marker}")

    while True:
        provider_choice = click.prompt(
            "\nEmbedding provider",
            default=str(current_config.embedding_provider),
            show_default=True,
        )
        try:
            embedding_provider = EmbeddingProvider(provider_choice)
            break
        except ValueError:
            click.echo(f"❌ Invalid provider: {provider_choice}")

    # Provider-specific configuration
    config_dict: Dict[str, Any] = {"embedding_provider": embedding_provider}

    if embedding_provider == EmbeddingProvider.OPENAI:
        config_dict["openai_api_key"] = click.prompt(
            "OpenAI API Key",
            default=current_config.openai_api_key,
            hide_input=True,
            show_default=False,
        )
        config_dict["openai_model"] = click.prompt(
            "OpenAI Model",
            default=current_config.openai_model,
            show_default=True,
        )
        # Use default base URL from current config without prompting to keep tests simple
        config_dict["openai_base_url"] = current_config.openai_base_url

    # Vector provider selection
    click.echo("\n🗄️ Select Vector Provider:")
    for i, provider in enumerate(get_available_vector_providers(), 1):
        default_marker = (
            " (current)" if provider == str(current_config.vector_provider) else ""
        )
        click.echo(f"  {i}. {provider}{default_marker}")

    while True:
        provider_choice = click.prompt(
            "\nVector provider",
            default=str(current_config.vector_provider),
            show_default=True,
        )
        try:
            vector_provider = VectorProvider(provider_choice)
            break
        except ValueError:
            click.echo(f"❌ Invalid provider: {provider_choice}")

    config_dict["vector_provider"] = vector_provider

    if vector_provider == VectorProvider.QDRANT_CLOUD:
        config_dict["qdrant_cloud_url"] = click.prompt(
            "Qdrant Cloud URL",
            default=current_config.qdrant_cloud_url,
            show_default=True,
        )
        config_dict["qdrant_api_key"] = click.prompt(
            "Qdrant Cloud API Key",
            default=current_config.qdrant_api_key,
            hide_input=True,
            show_default=False,
        )

    # Create new config and validate
    new_config = Config(**{**current_config.to_dict(), **config_dict})

    try:
        new_config.validate()
        click.echo("\n✅ Configuration is valid!")
    except ValueError as e:
        click.echo(f"\n❌ Configuration error: {e}")
        return

    # Show summary and confirm
    click.echo(f"\n📋 Configuration Summary:")
    click.echo(f"  Embedding Provider: {new_config.embedding_provider}")
    click.echo(f"  Vector Provider: {new_config.vector_provider}")

    if click.confirm("\nSave this configuration?"):
        click.echo("❌ Configuration saving to .env is disabled for security reasons")
    else:
        click.echo("❌ Configuration not saved")


@main.command(name="show-config")
def show_config() -> None:
    """Show current configuration."""
    try:
        config = Config.from_env()

        click.echo("🔧 Current Configuration\n")

        click.echo("📊 Embedding Provider:")
        click.echo(f"  Provider: {config.embedding_provider}")
        if config.embedding_provider == EmbeddingProvider.OPENAI:
            click.echo(f"  Model: {config.openai_model}")
            click.echo(f"  Base URL: {config.openai_base_url}")
            api_key_preview = (
                config.openai_api_key[:8] + "..."
                if config.openai_api_key
                else "Not set"
            )
            click.echo(f"  API Key: {api_key_preview}")

        click.echo("\n🗄️ Vector Provider:")
        click.echo(f"  Provider: {config.vector_provider}")
        if config.vector_provider == VectorProvider.QDRANT_CLOUD:
            click.echo(f"  URL: {config.qdrant_cloud_url}")
            api_key_preview = (
                config.qdrant_api_key[:8] + "..."
                if config.qdrant_api_key
                else "Not set"
            )
            click.echo(f"  API Key: {api_key_preview}")

        click.echo(f"\n⚙️ Other Settings:")
        click.echo(f"  Collection: {config.qdrant_collection}")
        click.echo(f"  Vector Size: {config.vector_size}")
        click.echo(f"  Max Tokens: {config.max_tokens}")

    except Exception as e:
        click.echo(f"❌ Failed to load configuration: {e}")


@main.command(name="test-providers")
@click.option("--embedding-provider", help="Embedding provider to test")
@click.option("--vector-provider", help="Vector provider to test")
@click.option("--openai-api-key", help="OpenAI API key")
@click.option("--qdrant-cloud-url", help="Qdrant Cloud URL")
@click.option("--qdrant-api-key", help="Qdrant Cloud API key")
def test_providers(
    embedding_provider: Optional[str] = None,
    vector_provider: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    qdrant_cloud_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
) -> None:
    """Test connectivity to specified providers."""
    click.echo("🔍 Testing Provider Connectivity\n")

    # Get base config
    config = Config.from_env()

    # Override with CLI options
    if embedding_provider:
        try:
            config.embedding_provider = EmbeddingProvider(embedding_provider)
        except ValueError:
            click.echo(f"❌ Invalid embedding provider: {embedding_provider}")
            return

    if vector_provider:
        try:
            config.vector_provider = VectorProvider(vector_provider)
        except ValueError:
            click.echo(f"❌ Invalid vector provider: {vector_provider}")
            return

    # Override credentials
    if openai_api_key:
        config.openai_api_key = openai_api_key
    if qdrant_cloud_url:
        config.qdrant_cloud_url = qdrant_cloud_url
    if qdrant_api_key:
        config.qdrant_api_key = qdrant_api_key

    try:
        config.validate()
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}")
        return

    factory = ProviderFactory()

    # Test embedding provider
    click.echo(f"📊 Testing {config.embedding_provider} embedding service...")
    try:
        embedding_config = config.get_embedding_service_config()
        embedding_service = factory.get_embedding_service(
            config.embedding_provider, embedding_config
        )

        if embedding_service.health_check():
            click.echo("  ✅ Embedding service is healthy")
        else:
            click.echo("  ❌ Embedding service health check failed")
    except Exception as e:
        click.echo(f"  ❌ Embedding service error: {e}")

    # Test vector provider
    click.echo(f"\n🗄️ Testing {config.vector_provider} vector service...")
    try:
        vector_config = config.get_vector_service_config()
        vector_service = factory.get_vector_service(
            config.vector_provider, vector_config
        )

        if vector_service.health_check():
            click.echo("  ✅ Vector service is healthy")
        else:
            click.echo("  ❌ Vector service health check failed")
    except Exception as e:
        click.echo(f"  ❌ Vector service error: {e}")

    click.echo("\n✅ Provider connectivity test completed")


@main.command(name="migrate")
@click.option(
    "--from-embedding", required=True, help="Source embedding provider (openai)"
)
@click.option(
    "--from-vector",
    required=True,
    help="Source vector provider (qdrant_cloud)",
)
@click.option(
    "--to-embedding", required=True, help="Target embedding provider (openai)"
)
@click.option(
    "--to-vector",
    required=True,
    help="Target vector provider (qdrant_cloud)",
)
@click.option("--batch-size", default=50, help="Migration batch size")
@click.option(
    "--backup/--no-backup", default=True, help="Create backup before migration"
)
@click.option("--dry-run", is_flag=True, help="Check compatibility without migrating")
@click.option(
    "--save-report/--no-save-report",
    default=False,
    help="Save migration report to file (default: no)",
)
@click.option("--openai-api-key", help="OpenAI API key for migration")
@click.option("--qdrant-cloud-url", help="Qdrant Cloud URL for migration")
@click.option("--qdrant-api-key", help="Qdrant Cloud API key for migration")
def migrate_providers(
    from_embedding: str,
    from_vector: str,
    to_embedding: str,
    to_vector: str,
    batch_size: int,
    backup: bool,
    dry_run: bool,
    save_report: bool,
    openai_api_key: Optional[str] = None,
    qdrant_cloud_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
) -> None:
    """Migrate data between different provider configurations.

    ⚠️  WARNING: This operation will transfer ALL vector data from source to target.
    Use --dry-run first to check compatibility without migrating.
    Always ensure you have backups before proceeding.
    """
    click.echo("🔄 Provider Migration Tool\n")

    try:
        # Prepare configuration overrides
        source_overrides = {}
        target_overrides = {}

        if openai_api_key:
            source_overrides["openai_api_key"] = openai_api_key
            target_overrides["openai_api_key"] = openai_api_key

        if qdrant_cloud_url:
            source_overrides["qdrant_cloud_url"] = qdrant_cloud_url
            target_overrides["qdrant_cloud_url"] = qdrant_cloud_url

        if qdrant_api_key:
            source_overrides["qdrant_api_key"] = qdrant_api_key
            target_overrides["qdrant_api_key"] = qdrant_api_key

        # Create migration configurations
        source_config, target_config = create_migration_config(
            from_embedding,
            from_vector,
            to_embedding,
            to_vector,
            source_overrides=source_overrides,
            target_overrides=target_overrides,
        )

        # Validate configurations
        try:
            source_config.validate()
            target_config.validate()
        except ValueError as e:
            click.echo(f"❌ Configuration error: {e}")
            return

        # Initialize migration manager
        migration_manager = MigrationManager(source_config, target_config)

        # Check compatibility
        click.echo("🔍 Checking provider compatibility...")
        compatibility = migration_manager.check_compatibility()

        click.echo(f"📊 Compatibility Report:")
        click.echo(
            f"  Embedding Compatible: {'✅' if compatibility.embedding_compatible else '❌'}"
        )
        click.echo(
            f"  Vector Compatible: {'✅' if compatibility.vector_compatible else '❌'}"
        )
        click.echo(
            f"  Dimension Match: {'✅' if compatibility.dimension_match else '❌'}"
        )

        if compatibility.source_dimensions > 0:
            click.echo(f"  Source Dimensions: {compatibility.source_dimensions}")
            click.echo(f"  Target Dimensions: {compatibility.target_dimensions}")

        if compatibility.warnings:
            click.echo("\n⚠️ Warnings:")
            for warning in compatibility.warnings:
                click.echo(f"  • {warning}")

        if not compatibility.fully_compatible:
            click.echo("\n❌ Providers are not fully compatible")
            if not click.confirm("Continue with migration anyway?"):
                click.echo("❌ Migration cancelled")
                return

        if dry_run:
            click.echo("\n✅ Dry run completed - no data migrated")
            return

        # Confirm migration with multiple safety checks
        click.echo(f"\n📋 Migration Plan:")
        click.echo(f"  From: {from_embedding}/{from_vector}")
        click.echo(f"  To: {to_embedding}/{to_vector}")
        click.echo(f"  Batch Size: {batch_size}")
        click.echo(f"  Backup: {'Yes' if backup else 'No'}")

        # First confirmation
        if not click.confirm(
            "\n⚠️  This will migrate ALL your vector data. Are you sure?"
        ):
            click.echo("❌ Migration cancelled")
            return

        # Second confirmation for safety
        confirmation_text = (
            f"{from_embedding}/{from_vector}-to-{to_embedding}/{to_vector}"
        )
        user_input = click.prompt(
            f"\n🔒 For safety, please type '{confirmation_text}' to confirm migration",
            type=str,
        )

        if user_input != confirmation_text:
            click.echo("❌ Migration cancelled - confirmation text did not match")
            return

        # Perform migration
        click.echo("\n🚀 Starting migration...")
        report = migration_manager.migrate_vectors(
            batch_size=batch_size, backup_first=backup
        )

        # Display results
        click.echo(f"\n📊 Migration Results:")
        click.echo(f"  Success Rate: {report.success_rate:.1f}%")
        click.echo(f"  Total Documents: {report.total_documents}")
        click.echo(f"  Migrated: {report.migrated_documents}")
        click.echo(f"  Failed: {report.failed_documents}")
        click.echo(f"  Duration: {report.duration:.2f} seconds")

        if report.performance_metrics:
            click.echo(f"\n⚡ Performance Metrics:")
            for metric, value in report.performance_metrics.items():
                click.echo(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

        if report.errors:
            click.echo(f"\n❌ Errors ({len(report.errors)}):")
            for error in report.errors[:5]:  # Show first 5 errors
                click.echo(f"  • {error}")
            if len(report.errors) > 5:
                click.echo(f"  ... and {len(report.errors) - 5} more errors")

        # Save migration report only if requested
        if save_report:

            report_file = f"migration_report_{int(report.start_time.timestamp())}.json"
            with open(report_file, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            click.echo(f"\n📄 Migration report saved to: {report_file}")
        else:
            click.echo(f"\n💬 Migration report not saved (use --save-report to save)")

        if report.success_rate >= 95:
            click.echo("\n🎉 Migration completed successfully!")
        elif report.success_rate >= 80:
            click.echo("\n⚠️ Migration completed with some issues")
        else:
            click.echo("\n❌ Migration completed with significant issues")

    except Exception as e:
        click.echo(f"❌ Migration failed: {e}")


@main.command(name="backup")
@click.option("--output", "-o", default="backup.json", help="Backup file path")
@click.option("--embedding-provider", help="Override embedding provider")
@click.option("--vector-provider", help="Override vector provider")
@click.option("--openai-api-key", help="Override OpenAI API key")
@click.option("--qdrant-cloud-url", help="Override Qdrant Cloud URL")
@click.option("--qdrant-api-key", help="Override Qdrant Cloud API key")
def create_backup(
    output: str,
    embedding_provider: Optional[str] = None,
    vector_provider: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    qdrant_cloud_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
) -> None:
    """Create backup of current vector collection."""
    click.echo("💾 Creating Vector Collection Backup\n")

    config = Config.from_env()

    # Apply CLI overrides
    if embedding_provider:
        try:
            config.embedding_provider = EmbeddingProvider(embedding_provider)
        except ValueError:
            click.echo(f"❌ Invalid embedding provider: {embedding_provider}")
            return

    if vector_provider:
        try:
            config.vector_provider = VectorProvider(vector_provider)
        except ValueError:
            click.echo(f"❌ Invalid vector provider: {vector_provider}")
            return

    if openai_api_key:
        config.openai_api_key = openai_api_key
    if qdrant_cloud_url:
        config.qdrant_cloud_url = qdrant_cloud_url
    if qdrant_api_key:
        config.qdrant_api_key = qdrant_api_key

    try:
        config.validate()
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}")
        return

    # Create dummy target config for migration manager
    migration_manager = MigrationManager(config, config)

    click.echo(f"📁 Backing up collection: {config.qdrant_collection}")
    click.echo(f"💾 Output file: {output}")

    result = migration_manager.create_backup(output)

    if result["success"]:
        click.echo(f"\n✅ Backup created successfully!")
        click.echo(f"  Points backed up: {result['points_backed_up']}")
        click.echo(f"  File size: {result['file_size'] / 1024 / 1024:.2f} MB")
        click.echo(f"  Backup file: {result['backup_path']}")
    else:
        click.echo(f"\n❌ Backup failed: {result['error']}")


@main.command(name="compare")
@click.option("--from-embedding", required=True, help="Source embedding provider")
@click.option("--from-vector", required=True, help="Source vector provider")
@click.option("--to-embedding", required=True, help="Target embedding provider")
@click.option("--to-vector", required=True, help="Target vector provider")
@click.option("--test-size", default=10, help="Number of test texts to use")
@click.option("--openai-api-key", help="OpenAI API key for comparison")
@click.option("--qdrant-cloud-url", help="Qdrant Cloud URL for comparison")
@click.option("--qdrant-api-key", help="Qdrant Cloud API key for comparison")
def compare_providers(
    from_embedding: str,
    from_vector: str,
    to_embedding: str,
    to_vector: str,
    test_size: int,
    openai_api_key: Optional[str] = None,
    qdrant_cloud_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
) -> None:
    """Compare performance between different provider configurations."""
    click.echo("⚡ Provider Performance Comparison\n")

    try:
        # Prepare configuration overrides
        overrides = {}
        if openai_api_key:
            overrides["openai_api_key"] = openai_api_key
        if qdrant_cloud_url:
            overrides["qdrant_cloud_url"] = qdrant_cloud_url
        if qdrant_api_key:
            overrides["qdrant_api_key"] = qdrant_api_key

        # Create configurations
        source_config, target_config = create_migration_config(
            from_embedding,
            from_vector,
            to_embedding,
            to_vector,
            source_overrides=overrides,
            target_overrides=overrides,
        )

        # Initialize migration manager
        migration_manager = MigrationManager(source_config, target_config)

        # Generate test texts
        test_texts = [
            f"This is test document number {i} for performance comparison."
            for i in range(test_size)
        ]

        click.echo(
            f"🧪 Running performance comparison with {test_size} test documents..."
        )
        comparison = migration_manager.compare_performance(test_texts)

        # Display results
        click.echo(f"\n📊 Performance Comparison Results:")
        click.echo(f"  Test Size: {comparison['test_size']} documents")

        if (
            "embedding_performance" in comparison
            and "error" not in comparison["embedding_performance"]
        ):
            emb_perf = comparison["embedding_performance"]
            click.echo(f"\n🔤 Embedding Performance:")
            click.echo(f"  Source ({comparison['source_provider']['embedding']}):")
            click.echo(f"    Total Time: {emb_perf['source']['total_time']:.4f}s")
            click.echo(
                f"    Avg per Text: {emb_perf['source']['avg_time_per_text']:.4f}s"
            )
            click.echo(f"    Dimensions: {emb_perf['source']['dimensions']}")

            click.echo(f"  Target ({comparison['target_provider']['embedding']}):")
            click.echo(f"    Total Time: {emb_perf['target']['total_time']:.4f}s")
            click.echo(
                f"    Avg per Text: {emb_perf['target']['avg_time_per_text']:.4f}s"
            )
            click.echo(f"    Dimensions: {emb_perf['target']['dimensions']}")

            speedup = emb_perf["speedup"]
            if speedup > 1:
                click.echo(f"  🚀 Target is {speedup:.2f}x faster")
            elif speedup < 1 and speedup > 0:
                click.echo(f"  🐌 Target is {1/speedup:.2f}x slower")

        if (
            "vector_performance" in comparison
            and "error" not in comparison["vector_performance"]
        ):
            vec_perf = comparison["vector_performance"]
            click.echo(f"\n🗄️ Vector Storage Performance:")
            click.echo(f"  Source ({comparison['source_provider']['vector']}):")
            click.echo(f"    Total Time: {vec_perf['source']['total_time']:.4f}s")
            click.echo(
                f"    Avg per Point: {vec_perf['source']['avg_time_per_point']:.4f}s"
            )

            click.echo(f"  Target ({comparison['target_provider']['vector']}):")
            click.echo(f"    Total Time: {vec_perf['target']['total_time']:.4f}s")
            click.echo(
                f"    Avg per Point: {vec_perf['target']['avg_time_per_point']:.4f}s"
            )

            speedup = vec_perf["speedup"]
            if speedup > 1:
                click.echo(f"  🚀 Target is {speedup:.2f}x faster")
            elif speedup < 1 and speedup > 0:
                click.echo(f"  🐌 Target is {1/speedup:.2f}x slower")

        # Save comparison report

        report_file = f"performance_comparison_{int(datetime.now().timestamp())}.json"
        with open(report_file, "w") as f:
            json.dump(comparison, f, indent=2)
        click.echo(f"\n📄 Comparison report saved to: {report_file}")

    except Exception as e:
        click.echo(f"❌ Performance comparison failed: {e}")


@main.command(name="config-templates")
@click.option("--tag", help="Filter templates by tag")
def list_config_templates(tag: Optional[str] = None) -> None:
    """List available configuration templates."""
    click.echo("📋 Configuration Templates\n")

    config_manager = ConfigurationManager()
    templates = config_manager.list_templates(tag=tag)

    if not templates:
        click.echo("No templates found")
        return

    for template in templates:
        click.echo(f"🔧 {template.name}")
        click.echo(f"   Description: {template.description}")
        click.echo(
            f"   Providers: {template.embedding_provider}/{template.vector_provider}"
        )

        if template.required_env_vars:
            click.echo(f"   Required vars: {', '.join(template.required_env_vars)}")

        if template.tags:
            click.echo(f"   Tags: {', '.join(template.tags)}")

        click.echo()


@main.command(name="config-from-template")
@click.argument("template_name")
@click.option("--validate-only", is_flag=True, help="Only validate, don't save")
@click.option("--output", "-o", help="Save configuration to file")
@click.option("--format", default="env", help="Output format (env, json, yaml)")
def create_config_from_template(
    template_name: str,
    validate_only: bool,
    output: Optional[str] = None,
    format: str = "env",
) -> None:
    """Create configuration from template."""
    click.echo(f"🔧 Creating Configuration from Template: {template_name}\n")

    config_manager = ConfigurationManager()

    # Load template
    template = config_manager.load_template(template_name)
    if not template:
        click.echo(f"❌ Template '{template_name}' not found")
        return

    click.echo(f"📋 Template: {template.description}")
    click.echo(
        f"🔧 Providers: {template.embedding_provider}/{template.vector_provider}"
    )

    # Check required environment variables
    missing_vars = []
    for var in template.required_env_vars:
        if var not in os.environ:
            missing_vars.append(var)

    if missing_vars:
        click.echo(f"\n❌ Missing required environment variables:")
        for var in missing_vars:
            click.echo(f"  • {var}")

        if click.confirm("\nSet these variables interactively?"):
            env_overrides = {}
            for var in missing_vars:
                if "key" in var.lower() or "token" in var.lower():
                    value = click.prompt(f"{var}", hide_input=True)
                else:
                    value = click.prompt(f"{var}")
                env_overrides[var.lower().replace("_", "_")] = value
                os.environ[var] = value
        else:
            click.echo("❌ Cannot proceed without required variables")
            return

    # Create configuration
    config = config_manager.create_config_from_template(template_name)
    if not config:
        click.echo("❌ Failed to create configuration from template")
        return

    # Validate configuration
    validation = config_manager.validate_config(config)

    click.echo(f"\n📊 Configuration Validation:")
    click.echo(f"  Valid: {'✅' if validation['valid'] else '❌'}")

    if validation["errors"]:
        click.echo(f"  Errors ({len(validation['errors'])}):")
        for error in validation["errors"]:
            click.echo(f"    • {error}")

    if validation["warnings"]:
        click.echo(f"  Warnings ({len(validation['warnings'])}):")
        for warning in validation["warnings"]:
            click.echo(f"    • {warning}")

    if validation["suggestions"]:
        click.echo(f"  Suggestions ({len(validation['suggestions'])}):")
        for suggestion in validation["suggestions"]:
            click.echo(f"    • {suggestion}")

    if not validation["valid"]:
        click.echo("\n❌ Configuration is invalid")
        return

    if validate_only:
        click.echo("\n✅ Configuration validation completed")
        return

    # Export configuration
    try:
        config_content = config_manager.export_config(
            config, format=format, include_secrets=True
        )

        if output:
            with open(output, "w") as f:
                f.write(config_content)
            click.echo(f"\n✅ Configuration saved to: {output}")
        else:
            click.echo(f"\n📄 Configuration ({format.upper()}):")
            click.echo("=" * 50)
            click.echo(config_content)
            click.echo("=" * 50)

    except Exception as e:
        click.echo(f"\n❌ Failed to export configuration: {e}")


@main.command(name="config-validate")
@click.option("--config-file", help="Configuration file to validate")
@click.option("--detailed", is_flag=True, help="Show detailed validation report")
def validate_config_file(
    config_file: Optional[str] = None, detailed: bool = False
) -> None:
    """Validate configuration file or current environment."""
    click.echo("🔍 Configuration Validation\n")

    if config_file:
        # Load configuration from file
        try:
            if config_file.endswith(".json"):
                with open(config_file, "r") as f:
                    config_dict = json.load(f)
                config = Config.from_dict(config_dict)
            else:
                # Assume it's an environment file
                click.echo("❌ Environment file validation not yet implemented")
                return
        except Exception as e:
            click.echo(f"❌ Failed to load configuration file: {e}")
            return
    else:
        # Use current environment
        try:
            config = Config.from_env()
        except Exception as e:
            click.echo(f"❌ Failed to load configuration from environment: {e}")
            return

    config_manager = ConfigurationManager()
    validation = config_manager.validate_config(config)

    # Display results
    click.echo(f"📊 Validation Results:")
    click.echo(f"  Status: {'✅ Valid' if validation['valid'] else '❌ Invalid'}")

    if validation["errors"]:
        click.echo(f"\n❌ Errors ({len(validation['errors'])}):")
        for i, error in enumerate(validation["errors"], 1):
            click.echo(f"  {i}. {error}")

    if validation["warnings"]:
        click.echo(f"\n⚠️ Warnings ({len(validation['warnings'])}):")
        for i, warning in enumerate(validation["warnings"], 1):
            click.echo(f"  {i}. {warning}")

    if validation["suggestions"]:
        click.echo(f"\n💡 Suggestions ({len(validation['suggestions'])}):")
        for i, suggestion in enumerate(validation["suggestions"], 1):
            click.echo(f"  {i}. {suggestion}")

    if detailed:
        click.echo(f"\n🔧 Configuration Details:")
        click.echo(f"  Embedding Provider: {config.embedding_provider}")
        click.echo(f"  Vector Provider: {config.vector_provider}")
        click.echo(f"  Vector Size: {config.vector_size}")
        click.echo(f"  Max Tokens: {config.max_tokens}")
        click.echo(f"  Collection: {config.qdrant_collection}")

        if config.embedding_provider == EmbeddingProvider.OPENAI:
            click.echo(f"  OpenAI Model: {config.openai_model}")
            api_key_masked = (
                config.openai_api_key[:8] + "..."
                if config.openai_api_key
                else "Not set"
            )
            click.echo(f"  OpenAI API Key: {api_key_masked}")

        if config.vector_provider == VectorProvider.QDRANT_CLOUD:
            click.echo(f"  Qdrant Cloud URL: {config.qdrant_cloud_url}")
            api_key_masked = (
                config.qdrant_api_key[:8] + "..."
                if config.qdrant_api_key
                else "Not set"
            )
            click.echo(f"  Qdrant API Key: {api_key_masked}")


@main.command(name="config-backup")
@click.option("--description", "-d", default="", help="Backup description")
@click.option("--config-file", help="Configuration file to backup")
def backup_config(description: str, config_file: Optional[str] = None) -> None:
    """Create configuration backup."""
    click.echo("💾 Creating Configuration Backup\n")

    if config_file:
        # Load from file
        try:
            with open(config_file, "r") as f:
                config_dict = json.load(f)
            config = Config.from_dict(config_dict)
        except Exception as e:
            click.echo(f"❌ Failed to load configuration file: {e}")
            return
    else:
        # Use current environment
        try:
            config = Config.from_env()
        except Exception as e:
            click.echo(f"❌ Failed to load configuration from environment: {e}")
            return

    config_manager = ConfigurationManager()
    backup_file = config_manager.create_backup(config, description)

    click.echo(f"✅ Configuration backup created:")
    click.echo(f"  File: {backup_file}")
    click.echo(f"  Description: {description or 'No description'}")
    click.echo(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


@main.command(name="config-backups")
def list_config_backups() -> None:
    """List configuration backups."""
    click.echo("📋 Configuration Backups\n")

    config_manager = ConfigurationManager()
    backups = config_manager.list_backups()

    if not backups:
        click.echo("No backups found")
        return

    for i, backup in enumerate(backups, 1):
        timestamp = datetime.fromisoformat(backup["timestamp"])
        size_mb = backup["size"] / 1024 / 1024

        click.echo(f"{i}. {Path(backup['file']).name}")
        click.echo(f"   Date: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        click.echo(f"   Size: {size_mb:.2f} MB")
        if backup["description"]:
            click.echo(f"   Description: {backup['description']}")
        click.echo()


@main.command(name="load-config")
@click.option("--config-file", required=True, help=".env file to load")
def load_config_file(config_file: str) -> None:
    """Load configuration from an .env file into environment."""
    try:
        try:
            from dotenv import load_dotenv

            load_dotenv(config_file, override=True)
        except Exception:
            # Fallback: simple parser for KEY=VALUE lines
            for line in Path(config_file).read_text().splitlines():
                if not line or line.strip().startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"')
        # Attempt to build config to validate loaded envs
        _ = Config.from_env()
        click.echo("✅ Configuration loaded successfully")
    except Exception as e:
        click.echo(f"❌ Failed to load configuration: {e}")


@main.command(name="config-import")
@click.option("--config-file", required=True, help="Path to JSON config file")
def import_config(config_file: str) -> None:
    """Import configuration from a JSON file and set environment variables."""
    try:
        with open(config_file, "r") as f:
            data = json.load(f)
        cfg = Config.from_dict(data)
        # Set env vars to reflect imported config
        os.environ["EMBEDDING_PROVIDER"] = cfg.embedding_provider.value
        os.environ["VECTOR_PROVIDER"] = cfg.vector_provider.value
        if cfg.embedding_provider == EmbeddingProvider.OPENAI:
            os.environ["OPENAI_API_KEY"] = cfg.openai_api_key
            os.environ["OPENAI_MODEL"] = cfg.openai_model
            os.environ["OPENAI_BASE_URL"] = cfg.openai_base_url
        if cfg.vector_provider == VectorProvider.QDRANT_CLOUD:
            os.environ["QDRANT_CLOUD_URL"] = cfg.qdrant_cloud_url
            os.environ["QDRANT_API_KEY"] = cfg.qdrant_api_key
        os.environ["COLLECTION_NAME"] = cfg.qdrant_collection
        os.environ["VECTOR_SIZE"] = str(cfg.vector_size)
        os.environ["MAX_TOKEN_LENGTH"] = str(cfg.max_tokens)
        os.environ["LOG_LEVEL"] = cfg.log_level
        os.environ["GIT_REPO_URL"] = cfg.repo_url
        os.environ["GIT_BRANCH"] = cfg.branch
        os.environ["MAX_WORKERS"] = str(cfg.max_workers)
        os.environ["CLOUDFLARE_ACCOUNT_ID"] = cfg.cloudflare_account_id
        os.environ["CLOUDFLARE_R2_ACCESS_KEY_ID"] = cfg.cloudflare_r2_access_key_id
        os.environ["CLOUDFLARE_R2_SECRET_ACCESS_KEY"] = (
            cfg.cloudflare_r2_secret_access_key
        )
        os.environ["CLOUDFLARE_R2_BUCKET"] = cfg.cloudflare_r2_bucket
        os.environ["CLOUDFLARE_R2_ENDPOINT"] = cfg.cloudflare_r2_endpoint
        os.environ["CLOUDFLARE_R2_KEY_PREFIX"] = cfg.cloudflare_r2_key_prefix
        os.environ["CLOUDFLARE_R2_SOFT_DELETE_ENABLED"] = str(
            cfg.cloudflare_r2_soft_delete_enabled
        ).lower()
        os.environ["CLOUDFLARE_R2_SOFT_DELETE_PREFIX"] = (
            cfg.cloudflare_r2_soft_delete_prefix
        )
        click.echo("✅ Configuration imported successfully")
    except Exception as e:
        click.echo(f"❌ Failed to import configuration: {e}")


@main.command(name="config-cleanup")
@click.option("--keep-days", default=30, help="Keep backups newer than N days")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted")
def cleanup_config_backups(keep_days: int, dry_run: bool) -> None:
    """Clean up old configuration backups."""
    click.echo(f"🧹 Configuration Backup Cleanup\n")

    config_manager = ConfigurationManager()

    if dry_run:
        click.echo(f"🔍 Dry run - showing backups older than {keep_days} days:")
        # List backups that would be deleted
        cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)

        old_backups = []
        for backup_file in config_manager.backups_dir.glob("config_backup_*.json"):
            if backup_file.stat().st_mtime < cutoff_time:
                old_backups.append(backup_file)

        if old_backups:
            for backup_file in old_backups:
                mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
                click.echo(
                    f"  • {backup_file.name} ({mtime.strftime('%Y-%m-%d %H:%M:%S')})"
                )
            click.echo(f"\nTotal: {len(old_backups)} backups would be deleted")
        else:
            click.echo("No old backups found")
    else:
        if click.confirm(f"Delete backups older than {keep_days} days?"):
            removed_count = config_manager.cleanup_old_backups(keep_days)
            click.echo(f"✅ Cleaned up {removed_count} old backups")
        else:
            click.echo("❌ Cleanup cancelled")


@main.command()
@click.option("--config-file", help="Path to configuration file")
@click.option("--embedding-provider", help="Override embedding provider")
@click.option("--vector-provider", help="Override vector provider")
@click.option("--openai-api-key", help="Override OpenAI API key")
@click.option("--qdrant-cloud-url", help="Override Qdrant Cloud URL")
@click.option("--qdrant-api-key", help="Override Qdrant Cloud API key")
def sync(
    config_file: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    vector_provider: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    qdrant_cloud_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
) -> None:
    """Perform incremental synchronization."""
    config = Config.from_env() if config_file is None else Config()

    # Apply CLI overrides
    if embedding_provider:
        try:
            config.embedding_provider = EmbeddingProvider(embedding_provider)
        except ValueError:
            raise click.BadParameter(
                f"Invalid embedding provider: {embedding_provider}"
            )

    if vector_provider:
        try:
            config.vector_provider = VectorProvider(vector_provider)
        except ValueError:
            raise click.BadParameter(f"Invalid vector provider: {vector_provider}")

    if openai_api_key:
        config.openai_api_key = openai_api_key
    if qdrant_cloud_url:
        config.qdrant_cloud_url = qdrant_cloud_url
    if qdrant_api_key:
        config.qdrant_api_key = qdrant_api_key

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        raise click.BadParameter(f"Configuration error: {e}")

    pipeline = SyncPipeline(config)

    # Health check first
    if not pipeline.health_check():
        click.echo(
            "❌ Health check failed. Please ensure Ollama and Qdrant are running."
        )
        return

    try:
        result = pipeline.sync()

        if result["status"] == "success":
            click.echo("✅ Sync completed successfully")
        elif result["status"] == "partial_success":
            click.echo("⚠️ Sync completed with some failures")

        if result["changes_detected"]:
            click.echo(
                f"📊 Changes: {result['upserted']} upserted, {result['deleted']} deleted, {result['renamed']} renamed"
            )
            if result["failed_files"]:
                click.echo(f"❌ Failed files: {', '.join(result['failed_files'])}")
        else:
            click.echo("📋 No changes detected")

        if result["status"] != "success":
            raise click.ClickException("Sync finished with errors")

    except SyncError as e:
        click.echo(f"❌ Sync failed: {e}")


@main.command(name="sync-cloudflare-r2")
@click.option("--config-file", help="Path to configuration file")
def sync_cloudflare_r2(config_file: Optional[str] = None) -> None:
    """Synchronize cleaned markdown documents to Cloudflare R2."""
    config = Config.from_env() if config_file is None else Config()

    try:
        config.validate_r2()
    except ValueError as e:
        raise click.BadParameter(f"Cloudflare R2 configuration error: {e}")

    pipeline = CloudflareR2SyncPipeline(config)

    try:
        result = pipeline.sync()
    except CloudflareR2SyncError as e:
        raise click.ClickException(f"Cloudflare R2 sync failed: {e}")

    if result["status"] == "success":
        click.echo("✅ Cloudflare R2 sync completed successfully")
    elif result["status"] == "partial_success":
        click.echo("⚠️ Cloudflare R2 sync completed with some failures")
    else:
        click.echo("❌ Cloudflare R2 sync failed")

    if result["changes_detected"]:
        click.echo(
            f"📦 Objects: {result['uploaded']} uploaded, {result['deleted']} deleted, {result['renamed']} renamed"
        )
        if result["failed_files"]:
            click.echo(f"❌ Failed paths: {', '.join(result['failed_files'])}")
    else:
        click.echo("📋 No markdown changes detected")

    if result["status"] != "success":
        raise click.ClickException("Cloudflare R2 sync finished with errors")


@main.command()
@click.option("--config-file", help="Path to configuration file")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
@click.option("--embedding-provider", help="Override embedding provider")
@click.option("--vector-provider", help="Override vector provider")
@click.option("--openai-api-key", help="Override OpenAI API key")
@click.option("--qdrant-cloud-url", help="Override Qdrant Cloud URL")
@click.option("--qdrant-api-key", help="Override Qdrant Cloud API key")
def reindex(
    config_file: Optional[str] = None,
    yes: bool = False,
    embedding_provider: Optional[str] = None,
    vector_provider: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    qdrant_cloud_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
) -> None:
    """Perform full reindexing (WARNING: This will recreate the collection)."""
    if not yes:
        if not click.confirm(
            "This will delete and recreate the entire collection. Continue?"
        ):
            click.echo("❌ Operation cancelled")
            return

    config = Config.from_env() if config_file is None else Config()

    # Apply CLI overrides
    if embedding_provider:
        try:
            config.embedding_provider = EmbeddingProvider(embedding_provider)
        except ValueError:
            click.echo(f"❌ Invalid embedding provider: {embedding_provider}")
            return

    if vector_provider:
        try:
            config.vector_provider = VectorProvider(vector_provider)
        except ValueError:
            click.echo(f"❌ Invalid vector provider: {vector_provider}")
            return

    if openai_api_key:
        config.openai_api_key = openai_api_key
    if qdrant_cloud_url:
        config.qdrant_cloud_url = qdrant_cloud_url
    if qdrant_api_key:
        config.qdrant_api_key = qdrant_api_key

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}")
        return

    pipeline = SyncPipeline(config)

    # Health check first
    if not pipeline.health_check():
        click.echo(
            "❌ Health check failed. Please ensure Ollama and Qdrant are running."
        )
        return

    try:
        result = pipeline.reindex_all()

        click.echo("✅ Reindex completed successfully")
        click.echo(f"📊 Processed {result['processed']}/{result['total_files']} files")

        if result["failed_files"]:
            click.echo(f"❌ Failed files: {', '.join(result['failed_files'])}")

    except SyncError as e:
        click.echo(f"❌ Reindex failed: {e}")


@main.command()
@click.option("--config-file", help="Path to configuration file")
@click.option("--embedding-provider", help="Override embedding provider")
@click.option("--vector-provider", help="Override vector provider")
@click.option("--openai-api-key", help="Override OpenAI API key")
@click.option("--qdrant-cloud-url", help="Override Qdrant Cloud URL")
@click.option("--qdrant-api-key", help="Override Qdrant Cloud API key")
@click.option("--verbose", is_flag=True, help="Show detailed provider status")
def health(
    config_file: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    vector_provider: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    qdrant_cloud_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Check health of all services."""
    config = Config.from_env() if config_file is None else Config()

    # Apply CLI overrides
    if embedding_provider:
        try:
            config.embedding_provider = EmbeddingProvider(embedding_provider)
        except ValueError:
            click.echo(f"❌ Invalid embedding provider: {embedding_provider}")
            return

    if vector_provider:
        try:
            config.vector_provider = VectorProvider(vector_provider)
        except ValueError:
            click.echo(f"❌ Invalid vector provider: {vector_provider}")
            return

    if openai_api_key:
        config.openai_api_key = openai_api_key
    if qdrant_cloud_url:
        config.qdrant_cloud_url = qdrant_cloud_url
    if qdrant_api_key:
        config.qdrant_api_key = qdrant_api_key

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}")
        return

    pipeline = SyncPipeline(config)

    click.echo("🔍 Checking service health...")

    if pipeline.health_check():
        click.echo("✅ All services are healthy")
        if verbose:
            click.echo(f"Embedding: {config.embedding_provider}")
            click.echo(f"Vector: {config.vector_provider}")
    else:
        click.echo("❌ Some services are not healthy")


@main.command(name="board-sync")
@click.option(
    "--board-idx",
    multiple=True,
    type=int,
    help="Specific board indices to ingest (repeatable)",
)
def board_sync(board_idx: tuple[int, ...]) -> None:
    """Ingest KNUE web board posts into Qdrant collection for boards."""
    click.echo("📰 KNUE Board Sync\n")

    config = Config.from_env()
    try:
        config.validate()
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}")
        return

    ingestor = KnueBoardIngestor(config)

    indices = board_idx if board_idx else config.board_indices
    click.echo(
        f"📚 Boards: {', '.join(str(i) for i in indices)} | Collection: {config.qdrant_board_collection}"
    )
    model_name = config.openai_model
    click.echo(f"🔤 Embedding: {config.embedding_provider} ({model_name})")

    try:
        result = ingestor.ingest(indices)
        click.echo("\n✅ Board sync completed")
        click.echo(
            f"  Processed: {result['processed']} | Deleted: {result['deleted']} | Upserted: {result['upserted']}"
        )
        if result["failed"]:
            click.echo(f"  Failed: {len(result['failed'])}")
            for link in result["failed"][:5]:
                click.echo(f"   - {link}")
    except Exception as e:
        click.echo(f"❌ Board sync failed: {e}")


@main.command(name="board-reindex")
@click.option(
    "--board-idx",
    multiple=True,
    type=int,
    help="Specific board indices to reindex (repeatable)",
)
@click.option(
    "--drop-collection/--no-drop-collection",
    default=None,
    help="Delete and recreate the entire board collection before reindexing",
)
def board_reindex(board_idx: tuple[int, ...], drop_collection: Optional[bool]) -> None:
    """Reindex KNUE web board posts into the board collection.

    If no board indices are provided and --drop-collection is not set, the command
    will default to dropping and recreating the collection. If board indices are
    provided and --drop-collection is not set, it will purge only those boards.
    """
    click.echo("📰 KNUE Board Reindex\n")

    config = Config.from_env()
    try:
        config.validate()
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}")
        return

    ingestor = KnueBoardIngestor(config)

    indices = board_idx if board_idx else config.board_indices
    # Determine default for drop_collection if not explicitly set
    if drop_collection is None:
        drop_collection = False if board_idx else True

    # Safety confirmation for destructive default
    if drop_collection and not board_idx:
        click.confirm(
            "⚠️  No specific boards provided. This will drop and reindex the ENTIRE collection. Continue?",
            abort=True,
        )

    click.echo(
        f"📚 Boards: {', '.join(str(i) for i in indices)} | Collection: {config.qdrant_board_collection}"
    )
    model_name = config.openai_model
    click.echo(f"🔤 Embedding: {config.embedding_provider} ({model_name})")
    click.echo(
        f"🧹 Drop collection: {'yes' if drop_collection else 'no'} | Mode: full ingest"
    )

    try:
        result = ingestor.reindex(indices, drop_collection=drop_collection)
        click.echo("\n✅ Board reindex completed")
        click.echo(
            f"  Processed: {result['processed']} | Deleted: {result['deleted']} | Upserted: {result['upserted']}"
        )
        if result["failed"]:
            click.echo(f"  Failed: {len(result['failed'])}")
            for link in result["failed"][:5]:
                click.echo(f"   - {link}")
    except Exception as e:
        click.echo(f"❌ Board reindex failed: {e}")


if __name__ == "__main__":
    main()
