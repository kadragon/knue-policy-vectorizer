---
id: SPEC-INDEX-MASTER
version: 1.0.0
scope: global
status: active
last-updated: 2025-10-16
owner: team-engineering
---

# Specification Index — KNUE Policy Vectorizer

## Active Specifications

### Canonical Roadmap
- **File**: `.spec/sync-pipeline/project-roadmap.spec.md`
- **ID**: SPEC-SYNC-ROADMAP-001
- **Scope**: 9-phase delivery roadmap with acceptance criteria and validation metrics
- **Status**: Complete (all phases 1-9 delivered as of 2025-10-15)

## Completed Phase Specifications (Archive)
These specs are preserved for reference and are superseded by `.spec/sync-pipeline/project-roadmap.spec.md`:
- Phase 1–9 outcomes documented in project-roadmap.spec.md

## Active Task Specifications

### Task: remove-env-export-feature
- **Status**: Completed 2025-10-15
- **Spec**: `.tasks/remove-env-export-feature/SPEC-DELTA.md`
- **Acceptance Criteria**:
  - No .env file generation from CLI
  - save_config_file function removed
  - config-export command removed
  - Tests updated, README updated
- **Validation**: All 281 tests passing

## Directory Structure
```
.spec/
├── SPEC_INDEX.md               # This file — master index
├── sync-pipeline/
│   └── project-roadmap.spec.md # Canonical 9-phase roadmap
├── tasks/                      # (Reserved for new task specs)
└── completed-phases/           # (Archive for historical specs)
```

## Specification Precedence
1. Canonical specs (`.spec/sync-pipeline/`) — highest authority
2. Task-specific specs (`.tasks/<task>/SPEC-DELTA.md`)
3. Policy/workflow specs (`.agents/`)
4. Implementation details (source code comments)

## Truth Contract
- All acceptance criteria in `.spec/` drive implementation and testing.
- Deviations from spec require spec update first, before code changes.
- Developers validate against spec requirements, not assumptions.
