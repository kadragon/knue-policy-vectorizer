# Proposals: PR 2 and PR 3

This document outlines two scoped, high‑impact PRs to be implemented after PR 1.

## PR 2 — Handle Git Renames and Type Changes in Sync Pipeline

**Overview**
- Ensure renamed markdown files are reflected correctly in Qdrant by deleting vectors for the old path and upserting for the new path. Treat type changes as modified.

**Problem**
- `GitWatcher.get_changed_files()` currently ignores `R` (rename) and `T` (type change). Renames lead to stale vectors for the old path and duplicate entries for the new path.

**Scope**
- In: Rename handling for `.md` files (excluding README*), type change as modified, compatibility with current added/modified/deleted behavior.
- Out: Non‑markdown files, advanced rename detection heuristics beyond Git diff, cross‑repo moves.

**Proposed Changes**
- `src/git_watcher.py`:
  - Detect `R` events; return both the old path (to delete) and the new path (to add) for markdown files.
  - Treat `T` events as modified.
  - Keep README* exclusion and sorting.
- `src/sync_pipeline.py`:
  - When a rename is reported, compute the old document ID from the old path and call `qdrant_service.delete_document_chunks(old_doc_id)` before upserting the new path.
  - Preserve existing behavior for A/M/D.

**Files**
- Modify: `src/git_watcher.py`, `src/sync_pipeline.py`
- Tests: `tests/test_git_watcher.py` (add rename/type‑change cases), `tests/test_sync_pipeline.py` (pipeline delete+upsert on rename)

**Test Plan**
- Unit: Mock Git diff entries for `R`, `T` and verify classification and pipeline actions.
- Ensure existing tests still pass.

**Risks & Mitigations**
- Git rename detection nuances: rely on GitPython diff flags; tests will mock diff entries. Document expected behavior.

**Acceptance Criteria**
- Renaming a `.md` file results in old vectors deleted and new vectors inserted; no duplicates remain.
- Type change events are treated as modified without regressions.

**Estimate**
- ~2–4 hours including tests.

---

## PR 3 — CI + Dev Tooling (uv, pytest, formatting checks)

**Overview**
- Add GitHub Actions CI to run unit tests and format checks consistently across Python versions.

**Problem**
- No CI pipeline; formatting tools (black/isort) are documented but not enforced. Quick regressions possible.

**Proposed Changes**
- `.github/workflows/ci.yml`:
  - Matrix: Python `3.9`, `3.10`, `3.11`.
  - Steps: checkout, install `uv`, `uv sync --frozen --group dev`, run `black --check`, `isort --check-only`, and `pytest -m "not integration and not slow"`.
- `pyproject.toml`:
  - Expand `dependency-groups.dev` to include `black`, `isort` (and optionally `mypy`).
  - Keep existing tool configs (black/isort/mypy) as is.

**Files**
- Add: `.github/workflows/ci.yml`
- Modify: `pyproject.toml` (dev group)

**Test Plan**
- CI should pass on a clean clone; run unit tests only by default. Integration tests can be triggered manually later.

**Risks & Mitigations**
- Formatting failures: CI will flag; contributors can run `uv run black src/ tests/` and `uv run isort src/ tests/` locally.

**Acceptance Criteria**
- CI green on main and PRs; unit tests run; formatting enforced.

**Estimate**
- ~1–2 hours including tuning the workflow.

---

## Implementation Order (Later)
1. PR 2: Rename/type‑change handling + tests.
2. PR 3: Add CI workflow + dev tooling group updates.

Each PR will be delivered in its own branch with focused commits and test updates.

