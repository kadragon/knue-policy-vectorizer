---
id: TASK-PLAN-CLOUDFLARE-R2-SYNC
version: 0.1.0
status: active
last-updated: 2025-10-14
owner: codex
---

# Implementation Plan — Cloudflare R2 Markdown Sync

## 1. Context & Reuse Inventory

- Existing repository mirror logic already implemented in `src/git_watcher.py` and coordinated through `src/sync_pipeline.py`; these components detect Markdown additions, updates, and deletions inside the cached Git repository under `repo_cache/`.
- Markdown cleanup utilities (frontmatter stripping, metadata extraction, UTF-8 handling) live in `src/markdown_processor.py` with unit coverage in `tests/test_markdown_processor.py`.
- Provider abstraction layer in `src/providers.py` and CLI entry points in `src/sync_pipeline.py` provide an extensible way to insert new sinks; current sink targets Qdrant.

## 2. High-Level Workflow

1. **Source Sync**  
   - Reuse `GitWatcher` to fetch and diff the upstream Git repository on each run.
   - Capture change events (added, modified, deleted Markdown files) to drive downstream processing.

2. **Document Preparation**  
   - Pass changed Markdown paths through `MarkdownProcessor` to normalize content, strip or reshape YAML/TOML frontmatter, and derive metadata (title, headings, tokens).
   - Introduce optional transformation hooks for Cloudflare-specific object metadata (e.g., custom HTTP headers, content-type).

3. **Cloudflare R2 Sync**  
   - Build a lightweight service wrapper (`src/cloudflare_r2_service.py`) for upload/delete operations using the S3-compatible API.
   - Map each normalized document to an R2 object (`key`, UTF-8 body, metadata headers) with optional soft-delete archive.

4. **Orchestration & CLI**  
   - Create a standalone pipeline (`src/r2_sync_pipeline.py`) that reuses `GitWatcher`/`MarkdownProcessor` but operates independently from Qdrant.
   - Add a dedicated CLI command (e.g., `sync-cloudflare-r2`) that invokes the new pipeline without touching the vector sync path.

5. **Observability & Resilience**  
   - Leverage existing structlog setup via `src/logger.py` for request/response logging, timing, and error visibility.
   - Ensure retry logic for transient networking issues with Cloudflare and surface failures without stopping the Qdrant path.

## 3. Detailed Tasks (TDD-Oriented)

1. **Requirements Capture**  
   - Finalize Cloudflare R2 bucket details, authentication (access key / secret key), and object naming conventions.  
   - Record in `.tasks/cloudflare-r2-sync/RESEARCH.md` once gathered.

2. **Pipeline Design**  
   - Define responsibilities of the standalone R2 pipeline, including change detection, upload/delete orchestration, and error handling.

3. **Test Scaffolding**  
   - Write unit tests for the new service and pipeline covering upload, delete, retry behaviour, and error handling.  
   - Mock S3-compatible HTTP calls to Cloudflare R2; ensure secrets are injected via environment variables defined in `.env.example`.

4. **Service Implementation**  
   - Implement `CloudflareR2Service` with upload/delete helpers, metadata serialization, and retry logic using `boto3`.

5. **Pipeline & CLI Wiring**  
   - Implement `CloudflareR2SyncPipeline` orchestrating GitWatcher → MarkdownProcessor → CloudflareR2Service.  
   - Register the new `sync-cloudflare-r2` CLI command alongside existing commands without altering the vector pipeline.

6. **Markdown Transformation Rules**  
   - Decide whether frontmatter is removed or reserialized; codify rules in helper functions under `markdown_processor` or a new `transformers/` module.  
   - Add targeted tests demonstrating the expected serialized payload (e.g., Markdown body + metadata) for typical policy docs.

7. **Environment & Configuration**  
   - Extend `.env.example` with Cloudflare R2 variables (account ID, access key ID, secret access key, bucket name, optional prefix, API endpoint).  
   - Document required configuration in `README.md` or a dedicated ops guide section.

8. **CLI & Scheduling**  
   - Add dedicated CLI command for R2 sync; document usage.  
   - If needed, adjust Docker compose or cron configuration to invoke the new command on schedule.

9. **Observability Enhancements**  
   - Integrate structured logging per operation (success/failure, latency).  
   - Optionally add metrics counters if the project maintains metrics.

10. **Validation & QA**  
    - Run full test suite (`uv run pytest -v`).  
    - Perform manual smoke test: run sync against a staging Cloudflare R2 bucket with sample Markdown files.  
    - Record results in `.tasks/cloudflare-r2-sync/PROGRESS.md`.

## 4. Risk & Mitigation

- **Rate Limits / API Quotas**: Implement exponential backoff and chunked uploads; consider multipart uploads for large files.  
- **Large Documents**: R2 follows S3 limits (single PUT up to 5 GiB, multipart up to 5 TiB); enforce preflight size checks and use multipart for large Markdown bundles if needed.  
- **Consistency**: Ensure that partial failures (e.g., network outage mid-run) are recoverable via rerun: rely on GitWatcher diffing and requeue on next sync.  
- **Credential Handling**: Use environment variables only; avoid writing secrets to disk.

## 5. Deliverables

- `src/cloudflare_r2_service.py` and supporting unit tests.  
- `src/r2_sync_pipeline.py` with CLI wiring (`sync-cloudflare-r2`) and accompanying tests.  
- Updated documentation (`README.md`, `.env.example`) and task artifacts (RESEARCH, SPEC-DELTA, PROGRESS).  
- Operations notes covering configuration, limits, and troubleshooting.
