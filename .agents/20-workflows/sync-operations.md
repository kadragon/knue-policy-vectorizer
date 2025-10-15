---
id: AG-WORKFLOW-KNUE-OPS-001
version: 1.0.0
scope: global
status: active
supersedes: []
depends: [AG-FOUND-KNUE-001, AG-POLICY-KNUE-DEV-001]
last-updated: 2025-10-15
owner: team-admin
---

# Sync Operations Workflow

## Core Components
- `git_watcher.GitWatcher`: monitors repository changes, including non-ASCII filenames, using diff-based polling.
- `markdown_processor.MarkdownProcessor`: strips front matter, derives titles and metadata, enforces token limits, and normalizes Korean text.
- `embedding_service.EmbeddingService`: wraps Ollama for batch `bge-m3` embeddings with validation and error handling.
- `qdrant_service.QdrantService`: manages collection lifecycle, batched CRUD operations, and similarity search.
- `sync_pipeline.SyncPipeline`: orchestrates end-to-end workflows, exposes CLI modes, and tracks commit checkpoints.
- `providers` module: centralizes provider configuration and overrides.
- `logger` module: establishes repository-wide `structlog` defaults.

## RSP-I Workflow
1. **Research** — capture hypotheses, metrics, and evidence in `.tasks/<task>/RESEARCH.md`.
2. **Spec** — translate findings into acceptance criteria via `.tasks/<task>/SPEC-DELTA.md` and `.spec/**`.
3. **Plan** — outline steps, dependencies, rollback, and validation in `.tasks/<task>/PLAN.md`.
4. **Implement** — run TDD (red → green → refactor) and update `.tasks/<task>/PROGRESS.md` as work advances.

## Testing Strategy
- Use pytest markers (`unit`, `integration`, `slow`) and maintain ≥95% overall coverage (100% on critical paths).
- Employ fixtures and `test_data/` assets for representative scenarios.
- Promote illustrative examples from specs into executable tests.
- Mock Ollama, Qdrant, and Git interactions for deterministic automation; avoid live network calls in tests.

## Deployment Paths
- **Development**: `docker-compose up qdrant` to run the database locally and execute the indexer manually.
- **Basic Production**: `docker-compose up` for indexer + Qdrant with loop-based scheduling.
- **Advanced Production**: `docker-compose -f docker-compose.cron.yml up` to enable cron scheduling and richer health checks.
- Keep environment variables consistent across `.env`, Docker secrets, and CI pipelines.

## Observability & Monitoring
- CLI health: `uv run python -m src.sync_pipeline health` or `docker-compose run --rm indexer ...` in container contexts.
- Direct probes: `curl http://localhost:6333/collections` (Qdrant) and `curl http://localhost:11434/api/version` (Ollama).
- Logs: follow `tail -f logs/vectorizer.log`, filter with `grep`, or use `docker-compose logs -f indexer`.
- Track metrics such as embedding latency, batch success rate, Qdrant upsert timings, memory footprint, and error frequency.

## Performance Tuning
- Defaults: `MAX_TOKEN_LENGTH=8192`, `MAX_DOCUMENT_CHARS=30000`, `BATCH_SIZE=10`, `MAX_WORKERS=4`.
- High-capacity profile: `MAX_WORKERS=12`, `BATCH_SIZE=25` with memory monitoring.
- Constrained profile: `MAX_WORKERS=2`, `BATCH_SIZE=5`, `MAX_TOKEN_LENGTH=4096`.
- Baseline benchmarks (100 docs): embeddings average 0.129 s/doc, storage 0.012 s/doc, batch mode ~2.12× faster, search latency <0.1 s.

## Troubleshooting
- **Ollama Connectivity**: verify `ollama serve`, confirm models via `ollama list`, and check network latency.
- **Qdrant Availability**: inspect `docker-compose ps qdrant` and hit REST health endpoints.
- **Memory Pressure**: lower `BATCH_SIZE` or `MAX_WORKERS` and observe `htop`/`docker stats`.
- **Storage Constraints**: monitor collection sizes, reclaim stale points, and ensure sufficient disk for Docker volumes.
- **Korean Text Handling**: confirm UTF-8 encoding via `file repo_cache/**/*.md` and validate locale settings.
