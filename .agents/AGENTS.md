---
id: AG-LOADER-KNUE-000
version: 1.1.0
scope: global
status: active
supersedes: []
depends: []
last-updated: 2025-10-16
owner: team-admin
---

# KNUE Policy Vectorizer — Agents Loader

> Load modular policies for the repository. Follow the declared order; higher numeric prefixes override earlier folders when conflicts arise.

## Load Order (with Folder Status)

| Prefix | Folder | Purpose | Status |
|--------|--------|---------|--------|
| 00 | foundations | Core principles, architecture, repository overview | ✅ Active |
| 10 | policies | Development standards, security, configuration | ✅ Active |
| 20 | workflows | Operational procedures, project tracking, testing | ✅ Active |
| 30 | roles | Team responsibilities, escalation paths, subagent behaviors | ✅ Active |
| 40 | templates | Standard formats, commit message, PR, task structures | ✅ Active |
| 90 | overrides | Emergency exceptions, temporary patches, incident responses | ✅ Active |

## Key Metadata Front Matter (Required)
```yaml
---
id: AG-<TYPE>-<DOMAIN>-<SEQ>
version: X.Y.Z
scope: global | folder:<path>
status: active | deprecated
supersedes: [AG-OLD-ID]
depends: [AG-DEP-ID, SPEC-ID]
last-updated: YYYY-MM-DD
owner: team-name
---
```

## Truth Hierarchy
1. **Specifications** (`.spec/`) — Canonical contracts
2. **Agents** (`.agents/`) — Policy & workflow guidance (this directory)
3. **Tasks** (`.tasks/`) — Implementation evidence and progress

## Load Rules
- All files must include required metadata front matter.
- Local folder-level `AGENTS.md` files may refine or override this order.
- Deprecated content should move to `.agents/_archive/YYYY/Q<N>/` with status `deprecated`.
- Conflicts resolved by numeric prefix (higher number wins).

## Related Indexes
- **Task Index**: `.tasks/TASKS_INDEX.md`
- **Spec Index**: `.spec/SPEC_INDEX.md`
- **Archive Index**: `.agents/_archive/INDEX.md` (when populated)
