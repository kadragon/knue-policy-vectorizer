---
id: TASK-SPEC-CLOUDFLARE-R2-SYNC
version: 1.0.0
status: active
last-updated: 2025-10-14
owner: codex
depends: [TASK-PLAN-CLOUDFLARE-R2-SYNC, TASK-RESEARCH-CLOUDFLARE-R2-SYNC]
---

# Spec Delta — Cloudflare R2 Markdown Sync

## 1. Scope

Synchronize Markdown documents pulled by the existing Git watch pipeline into a Cloudflare R2 bucket (`knue-vectorstore`) with cleaned content (frontmatter removed or normalized) and optional metadata headers. The sync must run alongside current Qdrant ingestion without degrading existing behavior.

## 2. Functional Requirements

1. **Document Discovery**
   - The system shall reuse `GitWatcher` change events to determine added, modified, or deleted Markdown files in `repo_cache/`.
   - Only files with `.md` or `.markdown` extensions are considered candidates by default; inclusion filters should be configurable.

2. **Content Normalization**
   - Each Markdown file shall be processed via `MarkdownProcessor` (or an extended variant) to remove YAML/TOML frontmatter and produce a cleaned body.
   - The original frontmatter, if present, shall be serialized into JSON metadata stored either as object metadata headers (prefix `x-amz-meta-`) or embedded within the uploaded object payload based on configuration.

3. **R2 Upload**
   - For each added or modified document, the system shall upload the normalized content to R2 using the configured bucket, access key, secret key, and endpoint.
   - Object keys shall default to the relative repository path (e.g., `policies/<path>.md`). A configurable prefix option must exist (e.g., `CLOUDFLARE_R2_KEY_PREFIX`).
   - Uploaded objects shall include `Content-Type: text/markdown; charset=utf-8`.

4. **Deletion Handling**
   - When a Markdown file is deleted upstream, the corresponding R2 object must be deleted within the same sync run unless soft-delete mode is enabled.
   - Soft-delete mode (if enabled via configuration) shall move deleted entries to a configurable prefix `deleted/` with a timestamp suffix.

5. **Error Handling & Retries**
   - Transient upload failures (network errors, 5xx responses) shall be retried up to 3 attempts before being marked as failed.
   - Failures shall not abort the entire pipeline; they must be logged with structured context (file path, error type) and surfaced via CLI exit status.

6. **Configuration**
   - Required environment variables: `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_R2_ACCESS_KEY_ID`, `CLOUDFLARE_R2_SECRET_ACCESS_KEY`, `CLOUDFLARE_R2_ENDPOINT`, `CLOUDFLARE_R2_BUCKET`.
   - Optional environment variables: `CLOUDFLARE_R2_KEY_PREFIX`, `CLOUDFLARE_R2_SOFT_DELETE_ENABLED`, retry/backoff tuning knobs.
   - `.env.example` must document all new variables with placeholder values.

7. **CLI Integration**
   - `uv run python -m src.sync_pipeline sync-cloudflare-r2` shall execute Git → Markdown → R2 flow independently of the vector sync.
   - The CLI shall exit with non-zero status if any upload/delete action ultimately fails.

## 3. Non-Functional Requirements

1. **Performance**
   - The R2 sync should handle at least 100 documents per run with total upload time under 2 minutes on baseline hardware (assuming average document size ≤ 200 KB).
   - Sequential processing is acceptable initially; batching/multipart uploads only required if individual objects exceed 5 MB.

2. **Reliability**
   - Idempotent behavior on rerun: re-uploading the same content should not generate duplicates or stale versions.
   - Failure logs must include enough detail to manually rerun affected paths.

3. **Security**
   - Secrets are injected via environment variables only; no secrets logged or committed.
   - S3 client must use HTTPS and verify SSL certificates.

4. **Observability**
   - Structured logs must record upload/delete outcomes, including object key, status, and duration.
   - Optional metrics hook for success/failure counts (if project adds metrics later); design should allow future extension.

## 4. Acceptance Criteria

1. Running the sync with new documents results in matching objects under the expected key paths in R2 with cleaned Markdown content (frontmatter removed) and correct content-type.
2. Deleting a document from the source repository removes or archives the corresponding R2 object according to configuration.
3. A simulated transient failure (mocked) retried up to the configured limit and surfaced as failure upon exhaustion.
4. CLI help output documents the new `sync-cloudflare-r2` command; `.env.example` lists all required configuration variables.
5. Pytest coverage includes unit tests for the R2 service and integration coverage for the standalone pipeline (GitWatcher → MarkdownProcessor → CloudflareR2Service) using mocks.
