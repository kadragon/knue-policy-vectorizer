"""Main synchronization pipeline for KNUE Policy Hub to Qdrant."""

import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import structlog

from config import Config
from embedding_service import EmbeddingService
from git_watcher import GitWatcher
from logger import setup_logger
from markdown_processor import MarkdownProcessor
from qdrant_service import QdrantService
from providers import (
    EmbeddingProvider, 
    VectorProvider, 
    ProviderFactory, 
    get_available_embedding_providers,
    get_available_vector_providers
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
    def embedding_service(self):
        """Get or create embedding service instance using provider factory."""
        if not hasattr(self, "_embedding_service"):
            factory = ProviderFactory()
            embedding_config = self.config.get_embedding_service_config()
            self._embedding_service = factory.get_embedding_service(
                self.config.embedding_provider,
                embedding_config
            )
        return self._embedding_service

    @property
    def qdrant_service(self):
        """Get or create vector service instance using provider factory."""
        if not hasattr(self, "_qdrant_service"):
            factory = ProviderFactory()
            vector_config = self.config.get_vector_service_config()
            # Add common configuration required by both local and cloud services
            vector_config.update({
                "collection_name": self.config.qdrant_collection,
                "vector_size": self.config.vector_size
            })
            self._qdrant_service = factory.get_vector_service(
                self.config.vector_provider,
                vector_config
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
        if not self.qdrant_service.collection_exists():
            self.logger.info(
                "Creating collection", collection=self.config.qdrant_collection
            )
            try:
                self.qdrant_service.create_collection()
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
                self.qdrant_service.delete_document_chunks(doc_id)
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
                        self.qdrant_service.upsert_point(
                            point_id=doc_id, vector=embedding, metadata=metadata
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
        self.qdrant_service.upsert_points_batch(batch_points)

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
            if self.qdrant_service.collection_exists():
                self.logger.info(
                    "Deleting existing collection", collection=collection_name
                )
                self.qdrant_service.delete_collection()

            self.logger.info("Creating new collection", collection=collection_name)
            try:
                self.qdrant_service.create_collection()
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
def main():
    """KNUE Policy Hub to Qdrant synchronization tool."""
    pass


@main.command(name="list-providers")
def list_providers():
    """List all available embedding and vector providers."""
    click.echo("🔧 Available Providers\n")
    
    click.echo("📊 Available Embedding Providers:")
    for provider in get_available_embedding_providers():
        click.echo(f"  • {provider}")
    
    click.echo("\n🗄️ Available Vector Providers:")
    for provider in get_available_vector_providers():
        click.echo(f"  • {provider}")


@main.command(name="configure")
def configure_providers():
    """Interactive configuration of embedding and vector providers."""
    click.echo("🔧 Multi-Provider Configuration\n")
    
    # Get current config as defaults
    current_config = Config.from_env()
    
    # Embedding provider selection
    click.echo("📊 Select Embedding Provider:")
    for i, provider in enumerate(get_available_embedding_providers(), 1):
        default_marker = " (current)" if provider == str(current_config.embedding_provider) else ""
        click.echo(f"  {i}. {provider}{default_marker}")
    
    while True:
        provider_choice = click.prompt(
            "\nEmbedding provider",
            default=str(current_config.embedding_provider),
            show_default=True
        )
        try:
            embedding_provider = EmbeddingProvider(provider_choice)
            break
        except ValueError:
            click.echo(f"❌ Invalid provider: {provider_choice}")
    
    # Provider-specific configuration
    config_dict = {"embedding_provider": embedding_provider}
    
    if embedding_provider == EmbeddingProvider.OPENAI:
        config_dict["openai_api_key"] = click.prompt(
            "OpenAI API Key",
            default=current_config.openai_api_key,
            hide_input=True,
            show_default=False
        )
        config_dict["openai_model"] = click.prompt(
            "OpenAI Model",
            default=current_config.openai_model,
            show_default=True
        )
        config_dict["openai_base_url"] = click.prompt(
            "OpenAI Base URL",
            default=current_config.openai_base_url,
            show_default=True
        )
    elif embedding_provider == EmbeddingProvider.OLLAMA:
        config_dict["ollama_url"] = click.prompt(
            "Ollama URL",
            default=current_config.ollama_url,
            show_default=True
        )
        config_dict["embedding_model"] = click.prompt(
            "Ollama Model",
            default=current_config.embedding_model,
            show_default=True
        )
    
    # Vector provider selection
    click.echo("\n🗄️ Select Vector Provider:")
    for i, provider in enumerate(get_available_vector_providers(), 1):
        default_marker = " (current)" if provider == str(current_config.vector_provider) else ""
        click.echo(f"  {i}. {provider}{default_marker}")
    
    while True:
        provider_choice = click.prompt(
            "\nVector provider",
            default=str(current_config.vector_provider),
            show_default=True
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
            show_default=True
        )
        config_dict["qdrant_api_key"] = click.prompt(
            "Qdrant Cloud API Key",
            default=current_config.qdrant_api_key,
            hide_input=True,
            show_default=False
        )
    elif vector_provider == VectorProvider.QDRANT_LOCAL:
        config_dict["qdrant_url"] = click.prompt(
            "Qdrant Local URL",
            default=current_config.qdrant_url,
            show_default=True
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
        # Generate .env content
        env_content = _generate_env_content(new_config)
        
        # Ask where to save
        save_path = click.prompt(
            "Save to file",
            default=".env",
            show_default=True
        )
        
        try:
            with open(save_path, 'w') as f:
                f.write(env_content)
            click.echo(f"✅ Configuration saved to {save_path}")
            click.echo("💡 Set these environment variables or source the file to use this configuration")
        except Exception as e:
            click.echo(f"❌ Failed to save configuration: {e}")
    else:
        click.echo("❌ Configuration not saved")


@main.command(name="show-config")
def show_config():
    """Show current configuration."""
    try:
        config = Config.from_env()
        
        click.echo("🔧 Current Configuration\n")
        
        click.echo("📊 Embedding Provider:")
        click.echo(f"  Provider: {config.embedding_provider}")
        if config.embedding_provider == EmbeddingProvider.OLLAMA:
            click.echo(f"  URL: {config.ollama_url}")
            click.echo(f"  Model: {config.embedding_model}")
        elif config.embedding_provider == EmbeddingProvider.OPENAI:
            click.echo(f"  Model: {config.openai_model}")
            click.echo(f"  Base URL: {config.openai_base_url}")
            api_key_preview = config.openai_api_key[:8] + "..." if config.openai_api_key else "Not set"
            click.echo(f"  API Key: {api_key_preview}")
        
        click.echo("\n🗄️ Vector Provider:")
        click.echo(f"  Provider: {config.vector_provider}")
        if config.vector_provider == VectorProvider.QDRANT_LOCAL:
            click.echo(f"  URL: {config.qdrant_url}")
        elif config.vector_provider == VectorProvider.QDRANT_CLOUD:
            click.echo(f"  URL: {config.qdrant_cloud_url}")
            api_key_preview = config.qdrant_api_key[:8] + "..." if config.qdrant_api_key else "Not set"
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
    qdrant_api_key: Optional[str] = None
):
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
            config.embedding_provider,
            embedding_config
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
            config.vector_provider,
            vector_config
        )
        
        if vector_service.health_check():
            click.echo("  ✅ Vector service is healthy")
        else:
            click.echo("  ❌ Vector service health check failed")
    except Exception as e:
        click.echo(f"  ❌ Vector service error: {e}")
    
    click.echo("\n✅ Provider connectivity test completed")


def _generate_env_content(config: Config) -> str:
    """Generate .env file content from configuration."""
    lines = [
        "# Multi-Provider Configuration",
        "# Generated by KNUE Policy Vectorizer",
        "",
        "# Provider Selection",
        f"EMBEDDING_PROVIDER={config.embedding_provider}",
        f"VECTOR_PROVIDER={config.vector_provider}",
        "",
    ]
    
    if config.embedding_provider == EmbeddingProvider.OPENAI:
        lines.extend([
            "# OpenAI Configuration",
            f"OPENAI_API_KEY={config.openai_api_key}",
            f"OPENAI_MODEL={config.openai_model}",
            f"OPENAI_BASE_URL={config.openai_base_url}",
            "",
        ])
    
    if config.embedding_provider == EmbeddingProvider.OLLAMA:
        lines.extend([
            "# Ollama Configuration",
            f"OLLAMA_URL={config.ollama_url}",
            f"OLLAMA_MODEL={config.embedding_model}",
            "",
        ])
    
    if config.vector_provider == VectorProvider.QDRANT_CLOUD:
        lines.extend([
            "# Qdrant Cloud Configuration",
            f"QDRANT_CLOUD_URL={config.qdrant_cloud_url}",
            f"QDRANT_API_KEY={config.qdrant_api_key}",
            "",
        ])
    
    if config.vector_provider == VectorProvider.QDRANT_LOCAL:
        lines.extend([
            "# Qdrant Local Configuration",
            f"QDRANT_URL={config.qdrant_url}",
            "",
        ])
    
    lines.extend([
        "# Common Settings",
        f"COLLECTION_NAME={config.qdrant_collection}",
        f"VECTOR_SIZE={config.vector_size}",
        f"MAX_TOKEN_LENGTH={config.max_tokens}",
        f"LOG_LEVEL={config.log_level}",
    ])
    
    return "\n".join(lines)


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
    qdrant_api_key: Optional[str] = None
):
    """Perform incremental synchronization."""
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

    except SyncError as e:
        click.echo(f"❌ Sync failed: {e}")


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
    qdrant_api_key: Optional[str] = None
):
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
def health(
    config_file: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    vector_provider: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    qdrant_cloud_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None
):
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
    else:
        click.echo("❌ Some services are not healthy")


if __name__ == "__main__":
    main()
