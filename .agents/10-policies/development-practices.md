---
id: AG-POLICY-KNUE-DEV-001
version: 1.0.0
scope: global
status: active
supersedes: []
depends: [AG-FOUND-KNUE-001]
last-updated: 2025-10-15
owner: team-admin
---

# Development Policies

## Environment & Setup
- Install dependencies with `uv sync`; perform editable installs via `uv pip install -e .`.
- Configure managed credentials instead of local services: set `OPENAI_API_KEY`, `OPENAI_MODEL`, `QDRANT_CLOUD_URL`, and `QDRANT_API_KEY` in your shell or secrets manager.
- Use `config/templates/openai-cloud.json` as the baseline template when bootstrapping new environments.

## Core Commands
- Formatting: `uv run black src/ tests/ && uv run isort src/ tests/`.
- Static analysis: `uv run mypy src/`.
- Test execution:
  - Full suite: `uv run pytest -v`.
  - Focused file: `uv run pytest tests/test_sync_pipeline.py -v`.
  - Coverage: `uv run pytest tests/ --cov=src/ --cov-report=html`.
  - Integration path: `uv run python scripts/test_full_sync_pipeline.py`.
- CLI operations: `uv run python -m src.sync_pipeline [health|sync|reindex|list-providers|show-config|configure|test-providers]`.

## Coding Standards
- Practice TDD: introduce a failing test, implement the minimal fix, then refactor on green.
- Align with `.spec/` as the authoritative contract (SDD); updates must reconcile with specs before merge.
- Style: PEP 8, Black (line length 88), isort `profile=black`, strict type hints, small cohesive functions.
- Logging: use `structlog`, include contextual metadata, and avoid sensitive data.
- Branch policy: branch from `main` using `feat/*`, `fix/*`, `docs/*`, `refactor/*`, or `chore/*`; commits follow `[Structural|Behavioral](scope) summary [task-slug]`.

## Security & Configuration
- Validate provider settings with `uv run python -m src.sync_pipeline test-providers` before production syncs.
- Regenerate embeddings via `reindex` whenever the embedding provider or vector dimensionality changes.
- Externalize secrets and environment variables; never commit `.env` files or unmasked credentials.
- Operate production at `LOG_LEVEL=INFO`; temporarily elevate to `DEBUG` only for short diagnostics.

## Test Fixtures & Sample Data
- Store test fixtures and sample data in `tests/fixtures/` following Python testing standards.
- Current fixtures include: `samples/` (markdown policy documents for integration testing and demos).
- Reference fixtures with relative paths from test files: `Path(__file__).parent.parent / "fixtures" / "samples" / "정책1_학사관리.md"`.

## Support & Maintenance
- Track roadmap updates in `.spec/sync-pipeline/project-roadmap.spec.md` and GitHub issues.
- Require code review with reproducible commands and relevant logs for each change.
- Update policy documents when architecture or process adjustments occur to preserve a single source of truth.
