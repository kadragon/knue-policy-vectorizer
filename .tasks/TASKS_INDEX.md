---
id: TASKS-INDEX-MASTER
version: 1.0.0
scope: global
status: active
last-updated: 2025-10-16
owner: team-admin
---

# Tasks Index — KNUE Policy Vectorizer

## Active Tasks

### remove-env-export-feature
- **Status**: ✅ Completed 2025-10-15
- **Goal**: Remove .env export functionality for security compliance
- **Artifacts**:
  - RESEARCH.md — Analysis of export mechanism and security concerns
  - SPEC-DELTA.md — Acceptance criteria and scope definition
  - PLAN.md — Implementation steps and rollback strategy
  - PROGRESS.md — Completion log and validation evidence
- **Validation**: All 281 tests passing; README updated; feature safely removed

## Archived Tasks

### cloudflare-r2-sync
- **Status**: ✅ Completed 2025-10-15
- **Location**: `.tasks/_archive/2025/Q4/cloudflare-r2-sync/`
- **Goal**: Implement S3-compatible backups to Cloudflare R2
- **Outcome**: boto3 integration, CloudflareR2Service, CLI command, all tests passing
- **Reference**: `.tasks/_archive/2025/Q4/ARCHIVE_INDEX.md`

### refactor-src-folder
- **Status**: ✅ Completed 2025-10-15 (minor lint cleanup pending)
- **Location**: `.tasks/_archive/2025/Q4/refactor-src-folder/`
- **Goal**: Reorganize src/ into logical service modules
- **Outcome**: Folder restructured, imports updated, functional tests passing
- **Reference**: `.tasks/_archive/2025/Q4/ARCHIVE_INDEX.md`

## Task Folder Structure
```
.tasks/
├── TASKS_INDEX.md              # This file — master index
├── remove-env-export-feature/  # Active task
│   ├── RESEARCH.md
│   ├── SPEC-DELTA.md
│   ├── PLAN.md
│   └── PROGRESS.md
└── _archive/
    └── 2025/Q4/
        ├── ARCHIVE_INDEX.md
        ├── cloudflare-r2-sync/
        └── refactor-src-folder/
```

## RSP-I Workflow
Every task follows the Research → Spec → Plan → Implement (RSPI) model:

1. **Research** (RESEARCH.md)
   - Analyze current state, hypotheses, metrics
   - Document findings and evidence
   - Identify risks and constraints

2. **Spec** (SPEC-DELTA.md)
   - Translate research into acceptance criteria
   - Define scope and non-functional requirements
   - Align with `.spec/` canonical specifications

3. **Plan** (PLAN.md)
   - Outline implementation steps
   - Identify dependencies and rollback paths
   - Estimate timeline and validation approach

4. **Implement** (PROGRESS.md)
   - Execute TDD (red → green → refactor)
   - Update progress log as work advances
   - Document validation evidence and blockers

## Task Status Definitions
- **Active**: Currently under investigation or implementation
- **Completed**: All acceptance criteria met; validation evidence collected
- **Archived**: Moved to `.tasks/_archive/YYYY/Q<N>/` after completion or deprecation
- **On-Hold**: Blocked by external dependencies or decisions
- **Deprecated**: Superseded by newer tasks or spec changes

## Archival Policy
Tasks are archived 30 days after completion or when superseded.
Archive location: `.tasks/_archive/YYYY/Q<quarter>/<task-name>/`
Index maintained in `.tasks/_archive/YYYY/Q<quarter>/ARCHIVE_INDEX.md`
