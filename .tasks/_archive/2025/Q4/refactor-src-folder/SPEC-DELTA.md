# Spec: Refactor src/ Folder Structure

## Goal
Refactor the monolithic src/ folder into modular subfolders to improve code organization, maintainability, and scalability.

## New Structure
- **src/services/**: External service integrations (cloudflare_r2_service.py, qdrant_service.py, qdrant_service_cloud.py, embedding_service_openai.py, knue_board_ingestor.py)
- **src/config/**: Configuration management (config.py, config_manager.py)
- **src/utils/**: Utility functions and helpers (crypto_utils.py, logger.py, markdown_processor.py, providers.py)
- **src/pipelines/**: Data processing pipelines (r2_sync_pipeline.py, sync_pipeline.py)
- **src/core/**: Core business logic (git_watcher.py, migration_tools.py)
- **src/__init__.py**: Package initialization (remains at root)

## Acceptance Criteria
- All source files moved to appropriate subfolders
- All import statements updated in src/ and tests/ (e.g., from src.config import Config → from src.config.config import Config)
- All tests pass after refactoring
- No breaking changes to public APIs or external interfaces
- Project runs without errors (pipelines, scripts functional)
- Documentation updated if references src paths