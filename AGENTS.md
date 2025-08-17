# KNUE Policy Vectorizer - Development Guide

## Project Overview

The KNUE Policy Vectorizer is a production-ready automation pipeline that synchronizes Korean policy documents from the KNUE Policy Hub GitHub repository to a Qdrant vector database. The system enables semantic search and RAG (Retrieval-Augmented Generation) capabilities for Korean university policy documents.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Host Machine  │    │  Docker Network │    │  Docker Network │
│                 │    │                 │    │                 │
│  Ollama:11434   │◄───┤  Indexer        │◄───┤  Qdrant:6333    │
│  bge-m3 model   │    │  (Python App)   │    │  (Vector DB)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Git Repository │
                    │  KNUE-Policy-Hub│
                    └─────────────────┘
```

## Technology Stack

- **Language**: Python 3.11+
- **Package Manager**: uv (modern Python package manager)
- **Vector Database**: Qdrant (1024-dimensional vectors)
- **Embedding Model**: bge-m3 via Ollama
- **Testing Framework**: pytest (104 tests, 100% critical path coverage)
- **Development Methodology**: Test-Driven Development (TDD)
- **Containerization**: Docker & Docker Compose
- **Logging**: structlog (structured logging)

## Project Structure & Module Organization

```
knue-policy-vectorizer/
├── src/                          # Core application code
│   ├── config.py                 # Configuration management
│   ├── git_watcher.py           # Git repository monitoring
│   ├── markdown_processor.py    # Markdown preprocessing
│   ├── embedding_service.py     # Ollama embedding generation
│   ├── qdrant_service.py        # Qdrant vector operations
│   ├── sync_pipeline.py         # Main orchestration pipeline (CLI + orchestration)
│   ├── providers.py             # Provider abstractions
│   └── logger.py                # Structured logging setup
├── tests/                        # Comprehensive test suite (104 tests)
│   └── test_*.py                # Pytest suite with markers (unit, integration, slow)
├── scripts/                      # Integration test scripts & utility scripts
│   └── test_full_sync_pipeline.py # Integration testing
├── config/                       # Configuration files & templates
├── test_data/                    # Sample inputs for testing
├── logs/                         # Runtime logs
├── repo_cache/                   # Working clone for watched repo
├── docker-compose.yml           # Main Docker setup (dev/prod)
├── docker-compose.cron.yml      # Advanced production setup
├── Dockerfile                   # Python application container
├── .env.example                 # Baseline env; copy to .env for local runs
├── README.md                    # User documentation
├── DOCKER.md                    # Docker deployment guide
└── TODO.md                      # Complete development history
```

## Build, Test, and Development Commands

### Installation & Setup

- Install (dev): `uv sync && uv pip install -e .`
- Local development setup:

  ```bash
  # Install dependencies
  uv sync

  # Install in development mode
  uv pip install -e .

  # Start services
  docker-compose up -d qdrant
  ollama serve
  ollama pull bge-m3
  ```

### Code Quality & Formatting

- Format: `uv run black src/ tests/ && uv run isort src/ tests/`
- Type check: `uv run mypy src/`

### Testing

- All tests: `uv run pytest -v`
- Single file: `uv run pytest tests/test_sync_pipeline.py -v`
- With coverage: `uv run pytest tests/ --cov=src/ --cov-report=html`
- Integration tests: `uv run python scripts/test_full_sync_pipeline.py`

### CLI Operations

- Run locally: `uv run python -m src.sync_pipeline health|sync|reindex`
- Provider tools: `uv run python -m src.sync_pipeline list-providers|show-config|configure|test-providers`
- Health check: `uv run python -m src.sync_pipeline health`
- Run sync: `uv run python -m src.sync_pipeline sync`

## Key Components

### 1. GitWatcher (`src/git_watcher.py`)

- Monitors Git repository for changes
- Detects added, modified, and deleted Markdown files
- Handles Korean UTF-8 filenames
- Implements efficient diff-based change detection

### 2. MarkdownProcessor (`src/markdown_processor.py`)

- Removes YAML/TOML frontmatter
- Extracts titles (H1 → filename → default fallback)
- Generates metadata following PRD schema
- Handles Korean text encoding
- Validates token length limits (8192 tokens for bge-m3)

### 3. EmbeddingService (`src/embedding_service.py`)

- Integrates with Ollama for bge-m3 embeddings
- Generates 1024-dimensional vectors
- Supports batch processing (2.12x performance improvement)
- Includes token validation and error handling
- Average performance: 0.129 seconds per embedding

### 4. QdrantService (`src/qdrant_service.py`)

- Manages Qdrant collections and points
- Implements CRUD operations with batch support
- Handles vector similarity search
- Includes comprehensive error handling
- Average storage time: 0.012 seconds per document

### 5. SyncPipeline (`src/sync_pipeline.py`)

- Orchestrates the entire pipeline
- Implements incremental synchronization
- Provides CLI interface (health, sync, reindex)
- Tracks commit history for change detection
- Includes structured logging and monitoring

## Development Guidelines

### Coding Style & Naming Conventions

- **Python Version**: Python 3.9+ (3.11+ recommended)
- **Formatting**: Black (88 cols) and isort (`profile=black`)
- **Type Hints**: Required; MyPy runs in strict mode — add annotations for new code
- **Naming Conventions**:
  - Modules/files: `snake_case.py`
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE`
- **Imports**: Ordered as stdlib, third‑party, local. Avoid unused; keep functions small and cohesive
- **Security**: Follow best practices; never log secrets; validate inputs

### Code Quality Standards

1. **Test-Driven Development (TDD)**

   - Write tests first, then implement functionality
   - Maintain 100% coverage for critical paths
   - Use pytest fixtures for consistent test data

2. **Code Style**

   - Follow Python PEP 8 guidelines
   - Use type hints throughout the codebase
   - Implement comprehensive error handling
   - Include docstrings for all public methods

3. **Logging**
   - Use structlog for all logging
   - Include contextual information (component, operation)
   - Log at appropriate levels (DEBUG, INFO, WARNING, ERROR)
   - Never log sensitive information

### Testing Guidelines

- **Framework**: Pytest. Place tests under `tests/` as `test_*.py`
- **Test Naming**: Use descriptive test names and `@pytest.mark.unit|integration|slow`
- **Test Strategy**: Prefer isolated unit tests with mocks; cover edge cases and error paths
- **Coverage**: Run `uv run pytest -v` before PRs; do not reduce coverage
- **Test Data**: Use `test_data/` where appropriate

### Testing Strategy

The project includes 104 comprehensive tests covering:

- **Unit Tests**: Individual component functionality
- **Integration Tests**: Component interaction
- **End-to-End Tests**: Complete pipeline workflows
- **Performance Tests**: Timing and efficiency validation

Key test files:

- `tests/test_git_watcher.py` (13 tests)
- `tests/test_markdown_processor.py` (17 tests)
- `tests/test_embedding_service.py` (20 tests)
- `tests/test_qdrant_service.py` (25 tests)
- `tests/test_sync_pipeline.py` (21 tests)

## Commit & Pull Request Guidelines

### Commit Standards

- **Format**: Use Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `optimize:` with optional scopes (e.g., `feat(config): ...`)
- **Style**: Imperative, concise subject; body explains motivation and effects; link issues
- **Security**: Never commit `.env` or keys. Start from `.env.example`

### Pull Request Requirements

- **Context**: Include what/why, repro commands, and logs/screenshots if UX/CLI output changes
- **Checklist**:
  - Tests added/updated
  - Docs touched (`README.md`/`DOCKER.md`)
  - `black/isort/mypy/pytest` all green
  - No secrets in diffs

## Security & Configuration

### Environment Management

- **Local Development**: Use `.env` file
- **Docker Environment**: Use `.env.docker` file
- **Production**: Use environment variables or custom `.env` files
- **Security**: Never commit `.env` or keys; validate provider settings before sync

### Key Environment Variables

```bash
# Git Repository
GIT_REPO_URL=https://github.com/KNUE-CS/KNUE-Policy-Hub.git
GIT_BRANCH=main

# Service URLs
QDRANT_URL=http://localhost:6333
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=bge-m3

# Processing Limits
MAX_TOKEN_LENGTH=8192
MAX_DOCUMENT_CHARS=30000
BATCH_SIZE=10
MAX_WORKERS=4

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/vectorizer.log
```

### Performance Tuning

**High-performance servers:**

```bash
export MAX_WORKERS=12
export BATCH_SIZE=25
export MAX_TOKEN_LENGTH=8192
```

**Low-resource environments:**

```bash
export MAX_WORKERS=2
export BATCH_SIZE=5
export MAX_TOKEN_LENGTH=4096
```

## Docker Deployment

### Available Configurations

1. **Development**: `docker-compose up qdrant`

   - Runs only Qdrant for local development
   - Allows manual sync testing

2. **Basic Production**: `docker-compose up`

   - Runs Qdrant + hourly auto-sync
   - Simple infinite loop scheduling

3. **Advanced Production**: `docker-compose -f docker-compose.cron.yml up`
   - Runs Qdrant + cron-based scheduling
   - More robust scheduling with health checks

## Performance Benchmarks

Current system performance (tested with 100 Korean policy documents):

- **Embedding Generation**: 0.129 seconds/document average
- **Vector Storage**: 0.012 seconds/document average
- **Batch Processing**: 2.12x faster than individual processing
- **Memory Usage**: ~512MB for 1000 documents
- **Search Performance**: <0.1 seconds for similarity search

## Monitoring and Observability

### Health Checks

```bash
# CLI health check
uv run python -m src.sync_pipeline health

# Docker health check
docker-compose run --rm indexer uv run python -m src.sync_pipeline health

# Direct service checks
curl http://localhost:6333/collections  # Qdrant
curl http://localhost:11434/api/version  # Ollama
```

### Logging

All components use structured logging with contextual information:

```python
logger.info("Processing document",
           document_id=doc_id,
           file_path=file_path,
           token_count=tokens)
```

### Metrics

Key metrics to monitor:

- Processing time per document
- Embedding generation success rate
- Qdrant storage success rate
- Memory usage trends
- Error frequency by component

## Debugging Common Issues

### 1. Ollama Connection Issues

- Verify Ollama is running: `curl http://localhost:11434/api/version`
- Check bge-m3 model: `ollama list`

### 2. Qdrant Connection Issues

- Check Qdrant status: `curl http://localhost:6333/collections`
- Verify Docker container: `docker-compose ps qdrant`

### 3. Memory Issues

- Reduce batch size: `export BATCH_SIZE=5`
- Lower token limit: `export MAX_TOKEN_LENGTH=4096`

### 4. Korean Text Issues

- Verify UTF-8 encoding: `file repo_cache/your-repo/*.md`
- Check locale settings

### Log Analysis

```bash
# View real-time logs
tail -f logs/vectorizer.log

# Filter by log level
grep "ERROR" logs/vectorizer.log

# Docker logs
docker-compose logs -f indexer
```

## Best Practices

### Code Development

1. **Always write tests first** (TDD methodology)
2. **Use type hints** for better code documentation
3. **Implement proper error handling** with specific exception types
4. **Add comprehensive logging** with structured context
5. **Follow the single responsibility principle** for each component

### Production Deployment

1. **Use cron-based scheduling** for production (`docker-compose.cron.yml`)
2. **Monitor system resources** (CPU, memory, disk, network)
3. **Implement proper backup strategies** for Qdrant data
4. **Set up alerting** for critical failures
5. **Regularly update dependencies** for security patches

### Performance Optimization

1. **Tune batch sizes** based on available memory
2. **Monitor embedding generation times** and adjust accordingly
3. **Use appropriate worker counts** for parallel processing
4. **Implement caching** where applicable
5. **Regular performance profiling** to identify bottlenecks

## Adding New Features

1. **Create tests first** (TDD approach)
2. **Implement minimal functionality** to pass tests
3. **Refactor and optimize** while maintaining test coverage
4. **Update documentation** as needed
5. **Run full test suite** before committing

## Future Development Areas

### Potential Enhancements

1. **Scalability Improvements**

   - Implement distributed processing
   - Add Redis for caching
   - Support multiple Git repositories

2. **Enhanced Monitoring**

   - Add Prometheus metrics
   - Implement alerting
   - Create performance dashboards

3. **Advanced Features**

   - Support for other document formats (PDF, DOCX)
   - Multi-language support
   - Advanced chunking strategies

4. **Integration Options**
   - REST API for external access
   - Webhook support for real-time updates
   - Integration with external monitoring systems

### Code Quality Improvements

1. **Additional Testing**

   - Property-based testing with Hypothesis
   - Load testing for high-volume scenarios
   - Chaos engineering for resilience

2. **Security Enhancements**
   - Add authentication for Qdrant access
   - Implement input validation
   - Add rate limiting

## Troubleshooting Guide

### Performance Issues

1. **Slow Embedding Generation**

   - Check Ollama GPU utilization: `ollama ps`
   - Verify network latency to Ollama
   - Consider reducing batch size

2. **Memory Consumption**

   - Monitor with `htop` or `docker stats`
   - Adjust `MAX_WORKERS` and `BATCH_SIZE`
   - Check for memory leaks in long-running processes

3. **Storage Issues**
   - Monitor Qdrant collection size
   - Check disk space for Docker volumes
   - Verify vector storage efficiency

## Security Tips

- **Configuration**: Validate provider settings before sync with `test-providers`
- **Provider Changes**: Changing embedding provider may require `reindex` due to vector dimension changes
- **Logging**: Avoid logging secrets; keep `LOG_LEVEL` appropriate for environment
- **Environment**: Never commit `.env` files; use `.env.example` as template

## Contact and Support

For development questions or issues:

1. **GitHub Issues**: Report bugs and feature requests
2. **Code Review**: All changes should be reviewed
3. **Documentation**: Keep this file updated with significant changes
4. **Testing**: Maintain test coverage above 95%

---

This guide should be updated whenever significant architectural changes are made to the system. The goal is to maintain a comprehensive resource for future development work while preserving the high-quality standards established during the initial implementation.
