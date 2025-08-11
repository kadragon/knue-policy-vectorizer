"""Main synchronization pipeline for KNUE Policy Hub to Qdrant."""
import click
import structlog
from typing import Dict, Any, List, Optional
from pathlib import Path

from config import Config
from git_watcher import GitWatcher
from markdown_processor import MarkdownProcessor
from embedding_service import EmbeddingService
from qdrant_service import QdrantService


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
        self.logger = logger.bind(pipeline="sync")
        
        # Simple state tracking for commits
        self._last_commit: Optional[str] = None
    
    @property
    def git_watcher(self) -> GitWatcher:
        """Get or create GitWatcher instance."""
        if not hasattr(self, '_git_watcher'):
            git_config = {
                'repo_url': self.config.repo_url,
                'branch': self.config.branch,
                'cache_dir': self.config.repo_cache_dir
            }
            self._git_watcher = GitWatcher(git_config)
        return self._git_watcher
    
    @property
    def markdown_processor(self) -> MarkdownProcessor:
        """Get or create MarkdownProcessor instance."""
        if not hasattr(self, '_markdown_processor'):
            self._markdown_processor = MarkdownProcessor()
        return self._markdown_processor
    
    @property
    def embedding_service(self) -> EmbeddingService:
        """Get or create EmbeddingService instance."""
        if not hasattr(self, '_embedding_service'):
            self._embedding_service = EmbeddingService(
                model_name=self.config.embedding_model,
                base_url=self.config.ollama_url,
                max_tokens=self.config.max_tokens
            )
        return self._embedding_service
    
    @property
    def qdrant_service(self) -> QdrantService:
        """Get or create QdrantService instance."""
        if not hasattr(self, '_qdrant_service'):
            # Parse URL to get host and port
            url_parts = self.config.qdrant_url.replace('http://', '').replace('https://', '')
            if ':' in url_parts:
                host, port = url_parts.split(':')
                port = int(port)
            else:
                host = url_parts
                port = 6333
            
            self._qdrant_service = QdrantService(
                host=host,
                port=port,
                collection_name=self.config.qdrant_collection,
                vector_size=self.config.vector_size
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
            self.logger.info("Creating collection", collection=self.config.qdrant_collection)
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
                # For deleted files, we only need the document ID
                doc_id = self.markdown_processor.calculate_document_id(
                    file_path, content=""
                )
                self.qdrant_service.delete_point(doc_id)
                return {
                    'file': file_path,
                    'action': 'delete',
                    'doc_id': doc_id,
                    'status': 'success'
                }
            
            else:
                # For add/modify, process the full pipeline
                content = self.git_watcher.get_file_content(file_path)
                
                # Process markdown
                processed = self.markdown_processor.process_markdown(content, file_path)
                
                # Check if content is valid (not too long)
                if not processed['is_valid']:
                    raise Exception(processed['validation_error'])
                
                # Generate document ID and metadata
                doc_id = self.markdown_processor.calculate_document_id(file_path, processed['content'])
                metadata = self.markdown_processor.generate_metadata(
                    processed['content'], 
                    processed['title'], 
                    processed['filename']
                )
                
                # Generate embedding
                embedding = self.embedding_service.generate_embedding(processed['content'])
                
                # Upsert to Qdrant
                self.qdrant_service.upsert_point(
                    point_id=doc_id,
                    vector=embedding,
                    payload=metadata
                )
                
                return {
                    'file': file_path,
                    'action': action,
                    'doc_id': doc_id,
                    'title': processed['title'],
                    'status': 'success'
                }
        
        except Exception as e:
            self.logger.error("Failed to process file", file=file_path, error=str(e))
            return {
                'file': file_path,
                'action': action,
                'status': 'failed',
                'error': str(e)
            }
    
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
                    'status': 'success',
                    'changes_detected': False,
                    'upserted': 0,
                    'deleted': 0,
                    'processed_files': [],
                    'deleted_files': [],
                    'failed_files': []
                }
            
            # Ensure collection exists
            if not self._ensure_collection_exists():
                raise SyncError("Failed to ensure collection exists")
            
            # Get changed files
            added_files, modified_files, deleted_files_list = self.git_watcher.get_changed_files(self._last_commit, current_commit)
            
            processed_files = []
            deleted_files = []
            failed_files = []
            
            # Process added and modified files
            for file_path in added_files + modified_files:
                if file_path.endswith('.md'):
                    result = self._process_file(file_path, 'upsert')
                    if result['status'] == 'success':
                        processed_files.append(result)
                    else:
                        failed_files.append(file_path)
            
            # Process deleted files
            for file_path in deleted_files_list:
                if file_path.endswith('.md'):
                    result = self._process_file(file_path, 'delete')
                    if result['status'] == 'success':
                        deleted_files.append(result)
                    else:
                        failed_files.append(file_path)
            
            # Determine final status
            if failed_files:
                status = 'partial_success'
            else:
                status = 'success'
            
            result = {
                'status': status,
                'changes_detected': True,
                'upserted': len(processed_files),
                'deleted': len(deleted_files),
                'processed_files': processed_files,
                'deleted_files': deleted_files,
                'failed_files': failed_files
            }
            
            # Update last commit
            self._last_commit = current_commit
            
            self.logger.info(
                "Sync completed",
                status=status,
                upserted=len(processed_files),
                deleted=len(deleted_files),
                failed=len(failed_files)
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
                    'status': 'success',
                    'total_files': 0,
                    'processed': 0,
                    'failed': 0,
                    'processed_files': [],
                    'failed_files': []
                }
            
            # Recreate collection (delete + create)
            collection_name = self.config.qdrant_collection
            if self.qdrant_service.collection_exists():
                self.logger.info("Deleting existing collection", collection=collection_name)
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
                result = self._process_file(file_path, 'upsert')
                if result['status'] == 'success':
                    processed_files.append(result)
                else:
                    failed_files.append(file_path)
            
            result = {
                'status': 'success',
                'total_files': len(files),
                'processed': len(processed_files),
                'failed': len(failed_files),
                'processed_files': processed_files,
                'failed_files': failed_files
            }
            
            self.logger.info(
                "Reindex completed",
                total=len(files),
                processed=len(processed_files),
                failed=len(failed_files)
            )
            
            return result
        
        except Exception as e:
            self.logger.error("Reindex operation failed", error=str(e))
            raise SyncError(f"Reindex operation failed: {str(e)}", cause=e)


@click.group()
def main():
    """KNUE Policy Hub to Qdrant synchronization tool."""
    pass


@main.command()
@click.option('--config-file', help='Path to configuration file')
def sync(config_file: Optional[str] = None):
    """Perform incremental synchronization."""
    config = Config.from_env() if config_file is None else Config()
    pipeline = SyncPipeline(config)
    
    # Health check first
    if not pipeline.health_check():
        click.echo("❌ Health check failed. Please ensure Ollama and Qdrant are running.")
        return
    
    try:
        result = pipeline.sync()
        
        if result['status'] == 'success':
            click.echo("✅ Sync completed successfully")
        elif result['status'] == 'partial_success':
            click.echo("⚠️ Sync completed with some failures")
        
        if result['changes_detected']:
            click.echo(f"📊 Changes: {result['upserted']} upserted, {result['deleted']} deleted")
            if result['failed_files']:
                click.echo(f"❌ Failed files: {', '.join(result['failed_files'])}")
        else:
            click.echo("📋 No changes detected")
        
    except SyncError as e:
        click.echo(f"❌ Sync failed: {e}")


@main.command()
@click.option('--config-file', help='Path to configuration file')
@click.option('--yes', is_flag=True, help='Skip confirmation prompt')
def reindex(config_file: Optional[str] = None, yes: bool = False):
    """Perform full reindexing (WARNING: This will recreate the collection)."""
    if not yes:
        if not click.confirm("This will delete and recreate the entire collection. Continue?"):
            click.echo("❌ Operation cancelled")
            return
    
    config = Config.from_env() if config_file is None else Config()
    pipeline = SyncPipeline(config)
    
    # Health check first
    if not pipeline.health_check():
        click.echo("❌ Health check failed. Please ensure Ollama and Qdrant are running.")
        return
    
    try:
        result = pipeline.reindex_all()
        
        click.echo("✅ Reindex completed successfully")
        click.echo(f"📊 Processed {result['processed']}/{result['total_files']} files")
        
        if result['failed_files']:
            click.echo(f"❌ Failed files: {', '.join(result['failed_files'])}")
        
    except SyncError as e:
        click.echo(f"❌ Reindex failed: {e}")


@main.command()
@click.option('--config-file', help='Path to configuration file')
def health(config_file: Optional[str] = None):
    """Check health of all services."""
    config = Config.from_env() if config_file is None else Config()
    pipeline = SyncPipeline(config)
    
    click.echo("🔍 Checking service health...")
    
    if pipeline.health_check():
        click.echo("✅ All services are healthy")
    else:
        click.echo("❌ Some services are not healthy")


if __name__ == "__main__":
    main()