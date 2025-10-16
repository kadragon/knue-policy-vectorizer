---
id: TASK-RESEARCH-CLOUDFLARE-R2-SYNC
version: 0.1.0
status: completed
last-updated: 2025-10-14
owner: codex
---

# Research Log — Cloudflare R2 Markdown Sync

## 1. Known Inputs

- **Repository Source**: KNUE Policy Hub mirror under `repo_cache/` (managed by existing `GitWatcher`).
- **Markdown Processing**: Current pipeline removes frontmatter and extracts metadata via `MarkdownProcessor`.
- **Deployment Context**: Python 3.11+ application orchestrated by `sync_pipeline` with provider abstraction.

## 2. Decisions to Confirm

1. **Target Repository Details** ✅
   - Git URL: https://github.com/kadragon/KNUE-Policy-Hub.git  
   - Branch: `main`  
   - Inclusion: repo-wide `.md` (excludes README via existing GitWatcher filters)

2. **Cloudflare R2 Configuration**
   - Account ID ✅ — provided via `.env` (`CLOUDFLARE_ACCOUNT_ID`)  
   - R2 bucket name ✅ — `knue-vectorstore` stored in `.env` (`CLOUDFLARE_R2_BUCKET`)  
   - Access key ID / secret key ✅ — configured as `CLOUDFLARE_R2_ACCESS_KEY_ID`, `CLOUDFLARE_R2_SECRET_ACCESS_KEY` in `.env`  
   - Endpoint ✅ — `https://6ed03d41ee9287a3e0e5bde9a6772812.r2.cloudflarestorage.com/knue-vectorstore` via `CLOUDFLARE_R2_ENDPOINT`  
   - Optional: R2 API token (S3 compatibility or Workers API) — confirm whether existing `CLOUDFLARE_API_TOKEN` suffices for access key management  
   - Environment variable naming convention ✅ — `.env.example` now mirrors runtime variables.
   - Soft-delete prefix ✅ — default `deleted/` exposed via `CLOUDFLARE_R2_SOFT_DELETE_PREFIX`

3. **Object Key Strategy** ✅
   - Relative repository path with forward slashes; optional prefix via `CLOUDFLARE_R2_KEY_PREFIX`  
   - Case preserved; no additional slugging  
   - Overwrite in place; optional soft-delete archive retains prior version

4. **Object Payload Structure** ✅
   - Body: cleaned Markdown (frontmatter removed)  
   - Frontmatter serialized to JSON and stored in object metadata header `frontmatter`

5. **Sync Frequency & Trigger** ✅
   - On-demand via `sync --push-cloudflare-r2`; reuse existing cron if present

6. **Error Handling & Observability** ✅
   - Structured logs plus CLI alerts; simple 3-attempt retry inside R2 service; CLI exit code 1 on failure

7. **Deletion Policy** ✅
   - Default: hard delete from R2; optional soft-delete archive via prefix/timestamp

## 3. Constraints & External Factors

- R2 follows S3 REST semantics; single PUT up to 5 GiB, multipart upload up to 5 TiB.  
- Egress/ingress costs apply; verify cost model for expected document volume.  
- Authentication uses access key ID/secret key or temporary credentials; store via environment variables only.

## 4. Data & Evidence Needed

- Cloudflare documentation links (API endpoints, limits, authentication).  
- Confirmation of Markdown formatting requirements from consuming service (e.g., RAG pipeline, web frontend).  
- Any existing Cloudflare integration patterns within the organization.

## 5. Next Actions

All research items incorporated into SPEC; no open tasks.
