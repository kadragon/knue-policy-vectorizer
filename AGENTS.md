# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Core code — `sync_pipeline.py` (CLI + orchestration), `markdown_processor.py`, `embedding_service*.py`, `qdrant_service*.py`, `git_watcher.py`, `config*.py`, `providers.py`, `logger.py`.
- `tests/`: Pytest suite (`test_*.py`) with markers (`unit`, `integration`, `slow`).
- `scripts/`: Utility/integration scripts (e.g., `test_full_sync_pipeline.py`, cron helpers).
- `config/`: Config templates/support files. `test_data/`: sample inputs. `logs/`: runtime logs. `repo_cache/`: working clone for watched repo.
- `.env.example`: baseline env; copy to `.env` for local runs.

## Build, Test, and Development Commands
- Install (dev): `uv sync && uv pip install -e .`
- Format: `uv run black src/ tests/ && uv run isort src/ tests/`
- Type check: `uv run mypy src/`
- Tests: `uv run pytest -v` (single file: `uv run pytest tests/test_sync_pipeline.py -v`).
- Run locally: `uv run python -m src.sync_pipeline health|sync|reindex`
- Provider tools: `uv run python -m src.sync_pipeline list-providers|show-config|configure|test-providers`

## Coding Style & Naming Conventions
- Python 3.9+; format with Black (88 cols) and isort (`profile=black`).
- Type hints required; MyPy runs in strict mode — add annotations for new code.
- Naming: modules/files `snake_case.py`; functions/vars `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE`.
- Imports ordered: stdlib, third‑party, local. Avoid unused; keep functions small and cohesive.

## Testing Guidelines
- Framework: Pytest. Place tests under `tests/` as `test_*.py`; use descriptive test names and `@pytest.mark.unit|integration|slow`.
- Prefer isolated unit tests with mocks; cover edge cases and error paths.
- Run `uv run pytest -v` before PRs; do not reduce coverage. Use `test_data/` where appropriate.

## Commit & Pull Request Guidelines
- Use Conventional Commits (seen in history): `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `optimize:` with optional scopes (e.g., `feat(config): ...`).
- Commits: imperative, concise subject; body explains motivation and effects; link issues.
- PRs: include context, what/why, repro commands, and logs/screenshots if UX/CLI output changes.
- Checklist: tests added/updated, docs touched (`README.md`/`DOCKER.md`), `black/isort/mypy/pytest` all green; no secrets in diffs.

## Security & Configuration Tips
- Never commit `.env` or keys. Start from `.env.example` or use `uv run --env-file .env.dev ...`.
- Validate provider settings before sync: `test-providers`. Changing embedding provider may require `reindex` due to vector dimension changes.
- Avoid logging secrets; keep `LOG_LEVEL` appropriate for environment.
