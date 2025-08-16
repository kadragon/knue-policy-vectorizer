"""
Migration tools for switching between providers in KNUE Policy Vectorizer.

This module provides utilities for:
- Migrating vectors between embedding providers
- Transferring data between vector database providers  
- Validating compatibility between providers
- Backup and restore functionality
- Performance comparison utilities
"""

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import structlog
from tqdm import tqdm

# Support both package and standalone imports
try:
    from .config import Config
    from .providers import (
        EmbeddingProvider,
        VectorProvider,
        ProviderFactory,
        EmbeddingServiceInterface,
        VectorServiceInterface,
    )
except Exception:  # pragma: no cover - fallback when imported as a script
    from config import Config  # type: ignore
    from providers import (  # type: ignore
        EmbeddingProvider,
        VectorProvider,
        ProviderFactory,
        EmbeddingServiceInterface,
        VectorServiceInterface,
    )

logger = structlog.get_logger(__name__)


@dataclass
class MigrationReport:
    """Report of migration operation results"""
    source_provider: str
    target_provider: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_documents: int = 0
    migrated_documents: int = 0
    failed_documents: int = 0
    errors: List[str] = None
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
    
    @property
    def success_rate(self) -> float:
        """Calculate migration success rate"""
        if self.total_documents == 0:
            return 0.0
        return (self.migrated_documents / self.total_documents) * 100
    
    @property
    def duration(self) -> float:
        """Calculate migration duration in seconds"""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export report as dictionary"""
        return {
            "source_provider": self.source_provider,
            "target_provider": self.target_provider,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_documents": self.total_documents,
            "migrated_documents": self.migrated_documents,
            "failed_documents": self.failed_documents,
            "success_rate": self.success_rate,
            "duration": self.duration,
            "errors": self.errors,
            "performance_metrics": self.performance_metrics
        }


@dataclass
class CompatibilityCheck:
    """Results of provider compatibility check"""
    embedding_compatible: bool
    vector_compatible: bool
    dimension_match: bool
    source_dimensions: int
    target_dimensions: int
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    @property
    def fully_compatible(self) -> bool:
        """Check if providers are fully compatible"""
        return self.embedding_compatible and self.vector_compatible and self.dimension_match


class MigrationManager:
    """Manager for provider migrations and compatibility checks"""
    
    def __init__(self, source_config: Config, target_config: Config):
        """Initialize migration manager with source and target configurations"""
        self.source_config = source_config
        self.target_config = target_config
        self.logger = logger.bind(component="MigrationManager")
        
        # Initialize provider factories
        self.factory = ProviderFactory()
        
        # Initialize services
        self._source_embedding = None
        self._target_embedding = None
        self._source_vector = None
        self._target_vector = None
    
    @property
    def source_embedding_service(self) -> EmbeddingServiceInterface:
        """Get source embedding service"""
        if self._source_embedding is None:
            config = self.source_config.get_embedding_service_config()
            self._source_embedding = self.factory.get_embedding_service(
                self.source_config.embedding_provider, config
            )
        return self._source_embedding
    
    @property
    def target_embedding_service(self) -> EmbeddingServiceInterface:
        """Get target embedding service"""
        if self._target_embedding is None:
            config = self.target_config.get_embedding_service_config()
            self._target_embedding = self.factory.get_embedding_service(
                self.target_config.embedding_provider, config
            )
        return self._target_embedding
    
    @property
    def source_vector_service(self) -> VectorServiceInterface:
        """Get source vector service"""
        if self._source_vector is None:
            config = self.source_config.get_vector_service_config()
            config.update({
                "collection_name": self.source_config.qdrant_collection,
                "vector_size": self.source_config.vector_size
            })
            self._source_vector = self.factory.get_vector_service(
                self.source_config.vector_provider, config
            )
        return self._source_vector
    
    @property
    def target_vector_service(self) -> VectorServiceInterface:
        """Get target vector service"""
        if self._target_vector is None:
            config = self.target_config.get_vector_service_config()
            config.update({
                "collection_name": self.target_config.qdrant_collection,
                "vector_size": self.target_config.vector_size
            })
            self._target_vector = self.factory.get_vector_service(
                self.target_config.vector_provider, config
            )
        return self._target_vector
    
    def check_compatibility(self) -> CompatibilityCheck:
        """Check compatibility between source and target providers"""
        self.logger.info("Checking provider compatibility")
        
        warnings = []
        
        # Check embedding provider compatibility
        embedding_compatible = True
        try:
            # Test embedding generation
            test_text = "test compatibility check"
            source_embedding = self.source_embedding_service.generate_embedding(test_text)
            target_embedding = self.target_embedding_service.generate_embedding(test_text)
            
            source_dims = len(source_embedding)
            target_dims = len(target_embedding)
            dimension_match = source_dims == target_dims
            
            if not dimension_match:
                warnings.append(
                    f"Dimension mismatch: source={source_dims}, target={target_dims}. "
                    "Migration will require re-embedding all documents."
                )
            
        except Exception as e:
            embedding_compatible = False
            source_dims = target_dims = 0
            dimension_match = False
            warnings.append(f"Embedding compatibility check failed: {e}")
        
        # Check vector service compatibility
        vector_compatible = True
        try:
            # Test vector service health
            source_healthy = self.source_vector_service.health_check()
            target_healthy = self.target_vector_service.health_check()
            
            if not source_healthy:
                vector_compatible = False
                warnings.append("Source vector service is not healthy")
            
            if not target_healthy:
                vector_compatible = False
                warnings.append("Target vector service is not healthy")
                
        except Exception as e:
            vector_compatible = False
            warnings.append(f"Vector service compatibility check failed: {e}")
        
        return CompatibilityCheck(
            embedding_compatible=embedding_compatible,
            vector_compatible=vector_compatible,
            dimension_match=dimension_match,
            source_dimensions=source_dims,
            target_dimensions=target_dims,
            warnings=warnings
        )
    
    def create_backup(self, backup_path: str) -> Dict[str, Any]:
        """Create backup of source collection"""
        self.logger.info("Creating backup", backup_path=backup_path)
        
        backup_data = {
            "created_at": datetime.now().isoformat(),
            "source_config": self.source_config.to_dict(),
            "collection_name": self.source_config.qdrant_collection,
            "points": []
        }
        
        try:
            # Get all points from source collection
            # Note: This is a simplified implementation
            # In practice, you'd need to implement pagination for large collections
            search_result = self.source_vector_service.search_points(
                self.source_config.qdrant_collection,
                [0.0] * self.source_config.vector_size,  # Dummy vector
                limit=10000  # Large limit to get all points
            )
            
            backup_data["points"] = search_result
            backup_data["total_points"] = len(search_result)
            
            # Save backup to file
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info("Backup created successfully", 
                           points=len(search_result), 
                           file=backup_path)
            
            return {
                "success": True,
                "backup_path": backup_path,
                "points_backed_up": len(search_result),
                "file_size": backup_file.stat().st_size
            }
            
        except Exception as e:
            self.logger.error("Backup creation failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    def restore_from_backup(self, backup_path: str) -> Dict[str, Any]:
        """Restore collection from backup"""
        self.logger.info("Restoring from backup", backup_path=backup_path)
        
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            points = backup_data["points"]
            collection_name = self.target_config.qdrant_collection
            
            # Create target collection if it doesn't exist
            if not self.target_vector_service.collection_exists(collection_name):
                self.target_vector_service.create_collection(
                    collection_name,
                    self.target_config.vector_size
                )
            
            # Restore points in batches
            batch_size = 100
            restored_count = 0
            
            for i in tqdm(range(0, len(points), batch_size), desc="Restoring points"):
                batch = points[i:i + batch_size]
                
                # Format points for upsert
                formatted_points = []
                for point in batch:
                    formatted_points.append({
                        "id": point["id"],
                        "vector": point.get("vector", []),
                        "payload": point.get("payload", {})
                    })
                
                self.target_vector_service.upsert_points(collection_name, formatted_points)
                restored_count += len(batch)
            
            self.logger.info("Restore completed successfully", 
                           points_restored=restored_count)
            
            return {
                "success": True,
                "points_restored": restored_count,
                "collection": collection_name
            }
            
        except Exception as e:
            self.logger.error("Restore failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    def migrate_vectors(self, batch_size: int = 50, backup_first: bool = True) -> MigrationReport:
        """Migrate vectors from source to target provider"""
        report = MigrationReport(
            source_provider=f"{self.source_config.embedding_provider}/{self.source_config.vector_provider}",
            target_provider=f"{self.target_config.embedding_provider}/{self.target_config.vector_provider}",
            start_time=datetime.now()
        )
        
        self.logger.info("Starting vector migration", 
                        source=report.source_provider,
                        target=report.target_provider)
        
        try:
            # Create backup if requested
            if backup_first:
                backup_path = f"backups/migration_backup_{int(time.time())}.json"
                backup_result = self.create_backup(backup_path)
                if not backup_result["success"]:
                    report.errors.append(f"Backup failed: {backup_result['error']}")
                    return report
                
                report.performance_metrics["backup_points"] = backup_result["points_backed_up"]
            
            # Check compatibility
            compatibility = self.check_compatibility()
            if not compatibility.fully_compatible:
                report.errors.extend(compatibility.warnings)
                if not compatibility.dimension_match:
                    self.logger.warning("Dimension mismatch detected, will re-embed documents")
            
            # Get source collection data
            # This is a simplified version - in practice you'd need pagination
            source_collection = self.source_config.qdrant_collection
            
            # Get all documents with their text content for re-embedding
            # Note: This requires the payload to contain the original text
            search_result = self.source_vector_service.search_points(
                source_collection,
                [0.0] * self.source_config.vector_size,
                limit=10000
            )
            
            report.total_documents = len(search_result)
            
            if report.total_documents == 0:
                self.logger.warning("No documents found in source collection")
                report.end_time = datetime.now()
                return report
            
            # Create target collection
            target_collection = self.target_config.qdrant_collection
            if not self.target_vector_service.collection_exists(target_collection):
                self.target_vector_service.create_collection(
                    target_collection,
                    self.target_config.vector_size
                )
            
            # Process documents in batches
            embedding_times = []
            storage_times = []
            
            for i in tqdm(range(0, len(search_result), batch_size), 
                         desc="Migrating documents"):
                batch = search_result[i:i + batch_size]
                
                try:
                    # Prepare texts for re-embedding
                    texts = []
                    batch_points = []
                    
                    for point in batch:
                        # Extract text content from payload
                        text_content = point.get("payload", {}).get("content", "")
                        if not text_content:
                            # Try alternative content fields
                            text_content = (
                                point.get("payload", {}).get("text", "") or
                                point.get("payload", {}).get("title", "") or
                                "No content available"
                            )
                        
                        texts.append(text_content)
                        batch_points.append(point)
                    
                    # Generate new embeddings if dimensions don't match
                    if not compatibility.dimension_match:
                        start_time = time.time()
                        embeddings = self.target_embedding_service.generate_embeddings_batch(texts)
                        embedding_times.append(time.time() - start_time)
                    else:
                        # Use existing embeddings
                        embeddings = [point.get("vector", []) for point in batch_points]
                    
                    # Prepare points for target vector service
                    target_points = []
                    for point, embedding in zip(batch_points, embeddings):
                        target_points.append({
                            "id": point["id"],
                            "vector": embedding,
                            "payload": point.get("payload", {})
                        })
                    
                    # Store in target vector service
                    start_time = time.time()
                    self.target_vector_service.upsert_points(target_collection, target_points)
                    storage_times.append(time.time() - start_time)
                    
                    report.migrated_documents += len(batch)
                    
                except Exception as e:
                    error_msg = f"Batch {i//batch_size + 1} failed: {str(e)}"
                    report.errors.append(error_msg)
                    report.failed_documents += len(batch)
                    self.logger.error("Batch migration failed", 
                                    batch=i//batch_size + 1, 
                                    error=str(e))
            
            # Calculate performance metrics
            if embedding_times:
                report.performance_metrics["avg_embedding_time"] = sum(embedding_times) / len(embedding_times)
                report.performance_metrics["total_embedding_time"] = sum(embedding_times)
            
            if storage_times:
                report.performance_metrics["avg_storage_time"] = sum(storage_times) / len(storage_times)
                report.performance_metrics["total_storage_time"] = sum(storage_times)
            
            report.end_time = datetime.now()
            
            self.logger.info("Migration completed", 
                           success_rate=report.success_rate,
                           migrated=report.migrated_documents,
                           failed=report.failed_documents,
                           duration=report.duration)
            
        except Exception as e:
            report.errors.append(f"Migration failed: {str(e)}")
            report.end_time = datetime.now()
            self.logger.error("Migration failed", error=str(e))
        
        return report
    
    def compare_performance(self, test_texts: List[str]) -> Dict[str, Any]:
        """Compare performance between source and target providers"""
        self.logger.info("Starting performance comparison")
        
        comparison = {
            "test_size": len(test_texts),
            "source_provider": {
                "embedding": str(self.source_config.embedding_provider),
                "vector": str(self.source_config.vector_provider)
            },
            "target_provider": {
                "embedding": str(self.target_config.embedding_provider),
                "vector": str(self.target_config.vector_provider)
            },
            "embedding_performance": {},
            "vector_performance": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Test embedding performance
        try:
            # Source embedding performance
            start_time = time.time()
            source_embeddings = self.source_embedding_service.generate_embeddings_batch(test_texts)
            source_embedding_time = time.time() - start_time
            
            # Target embedding performance  
            start_time = time.time()
            target_embeddings = self.target_embedding_service.generate_embeddings_batch(test_texts)
            target_embedding_time = time.time() - start_time
            
            comparison["embedding_performance"] = {
                "source": {
                    "total_time": source_embedding_time,
                    "avg_time_per_text": source_embedding_time / len(test_texts),
                    "dimensions": len(source_embeddings[0]) if source_embeddings else 0
                },
                "target": {
                    "total_time": target_embedding_time,
                    "avg_time_per_text": target_embedding_time / len(test_texts),
                    "dimensions": len(target_embeddings[0]) if target_embeddings else 0
                },
                "speedup": source_embedding_time / target_embedding_time if target_embedding_time > 0 else 0
            }
            
        except Exception as e:
            comparison["embedding_performance"]["error"] = str(e)
        
        # Test vector storage performance (if collections exist)
        try:
            source_collection = self.source_config.qdrant_collection
            target_collection = self.target_config.qdrant_collection
            
            if (self.source_vector_service.collection_exists(source_collection) and 
                self.target_vector_service.collection_exists(target_collection)):
                
                # Prepare test points
                test_points = []
                for i, (text, embedding) in enumerate(zip(test_texts, source_embeddings)):
                    test_points.append({
                        "id": f"perf_test_{i}",
                        "vector": embedding,
                        "payload": {"content": text, "test": True}
                    })
                
                # Source storage performance
                start_time = time.time()
                self.source_vector_service.upsert_points(source_collection, test_points)
                source_storage_time = time.time() - start_time
                
                # Target storage performance
                start_time = time.time()
                self.target_vector_service.upsert_points(target_collection, test_points)
                target_storage_time = time.time() - start_time
                
                comparison["vector_performance"] = {
                    "source": {
                        "total_time": source_storage_time,
                        "avg_time_per_point": source_storage_time / len(test_points)
                    },
                    "target": {
                        "total_time": target_storage_time,
                        "avg_time_per_point": target_storage_time / len(test_points)
                    },
                    "speedup": source_storage_time / target_storage_time if target_storage_time > 0 else 0
                }
                
                # Clean up test points
                test_ids = [point["id"] for point in test_points]
                self.source_vector_service.delete_points(source_collection, test_ids)
                self.target_vector_service.delete_points(target_collection, test_ids)
                
        except Exception as e:
            comparison["vector_performance"]["error"] = str(e)
        
        return comparison


def create_migration_config(
    source_embedding: str,
    source_vector: str, 
    target_embedding: str,
    target_vector: str,
    **kwargs
) -> Tuple[Config, Config]:
    """Create source and target configurations for migration"""
    
    # Base configuration
    base_config = Config.from_env()
    
    # Source configuration (copy to avoid shared dict from mocks)
    source_config_dict = dict(base_config.to_dict())
    source_config_dict.update({
        "embedding_provider": EmbeddingProvider(source_embedding),
        "vector_provider": VectorProvider(source_vector)
    })
    source_config_dict.update(kwargs.get('source_overrides', {}))
    source_config = Config.from_dict(source_config_dict)
    
    # Target configuration (separate copy)
    target_config_dict = dict(base_config.to_dict())
    target_config_dict.update({
        "embedding_provider": EmbeddingProvider(target_embedding),
        "vector_provider": VectorProvider(target_vector)
    })
    target_config_dict.update(kwargs.get('target_overrides', {}))
    target_config = Config.from_dict(target_config_dict)
    
    return source_config, target_config
