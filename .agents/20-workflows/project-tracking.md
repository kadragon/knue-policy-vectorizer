---
id: AG-WORKFLOW-KNUE-TRACK-001
version: 1.0.0
scope: global
status: active
supersedes: []
depends: [AG-FOUND-KNUE-001, SPEC-SYNC-ROADMAP-001]
last-updated: 2025-10-15
owner: team-engineering
---

# Project Tracking Workflow

## Status Snapshot (2025-10-15)
- Refer to `.spec/sync-pipeline/project-roadmap.spec.md` for the authoritative ledger of phases 1–9 and associated artefacts.
- All roadmap phases are complete; repository contains validated code, tests, GitHub Actions automation, and operator documentation matching the spec.
- Quality gates achieved: 104 core tests + 85 multi-provider tests, latency benchmarks (0.129 s/embedding, 0.012 s/Qdrant write), UTF-8 Korean coverage.

## Maintenance Routine
- When new scope is introduced, update the spec first (acceptance criteria, deliverables, validation evidence), then mirror high-level impacts here.
- Ensure RSP-I artefacts under `.tasks/` are refreshed (RESEARCH → SPEC-DELTA → PLAN → PROGRESS) alongside spec updates.
- Keep CLI and environment command references synchronized with the spec; verify changes via `uv run pytest` and relevant scripts before marking phases complete.

## Validation Checklist
- Confirm no-op sync behaviour prior to release freezes.
- Audit provider configuration (`configure`, `list-providers`, `show-config`, `test-providers`) whenever new environments are added.
- Re-run migration tooling and backup procedures before switching embedding/vector providers.
- For scheduled operations, validate the GitHub Actions workflow (`.github/workflows/daily-r2-sync.yml`) and confirm Cloudflare R2 backups execute successfully.

## Communication
- Replace ad-hoc TODO lists with updates to this workflow file and the spec to maintain a single source of truth.
- Surface deviations or blockers in `.tasks/<task>/PROGRESS.md` and link back to the relevant spec section for traceability.
