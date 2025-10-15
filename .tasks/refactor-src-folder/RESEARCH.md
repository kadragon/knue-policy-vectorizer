# Research: Refactor src/ Folder Structure

## Current Structure
- Total files in src/: 16
- Files: __init__.py, cloudflare_r2_service.py, config.py, config_manager.py, crypto_utils.py, embedding_service_openai.py, git_watcher.py, knue_board_ingestor.py, logger.py, markdown_processor.py, migration_tools.py, providers.py, qdrant_service.py, qdrant_service_cloud.py, r2_sync_pipeline.py, sync_pipeline.py

## Dependencies Analysis
- **tests/**: 19 import statements from src modules (e.g., from src.config import Config, from src.providers import ...)
- **scripts/**: No direct imports from src
- **.spec/**: No direct imports from src
- **.agents/**: No direct imports from src

## Impact Assessment
- Primary impact on tests/ (all test files import src modules)
- scripts/, .spec/, .agents/ unaffected directly, but may need updates if they reference src paths
- Potential breaking changes: All imports in tests/ and any external references need updating

## Proposed New Structure
- **services/**: cloudflare_r2_service.py, qdrant_service.py, qdrant_service_cloud.py, embedding_service_openai.py, knue_board_ingestor.py
- **config/**: config.py, config_manager.py
- **utils/**: crypto_utils.py, logger.py, markdown_processor.py, providers.py
- **pipelines/**: r2_sync_pipeline.py, sync_pipeline.py
- **core/**: git_watcher.py, migration_tools.py
- **__init__.py**: Remains at root

## Hypotheses
- Modular structure will improve maintainability
- Import updates will be straightforward (change src.module to src.subfolder.module)
- Tests will need corresponding import updates