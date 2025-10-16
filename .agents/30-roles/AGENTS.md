---
id: AG-ROLES-KNUE-001
version: 1.0.0
scope: global
status: active
supersedes: []
depends: [AG-POLICY-KNUE-DEV-001]
last-updated: 2025-10-16
owner: team-admin
---

# Role-Based Responsibilities

## Engineering Team
- **Code Review**: Mandate peer review on all feature branches before merge.
- **Testing**: Maintain ≥95% test coverage on critical paths; promote TDD discipline in PRs.
- **Documentation**: Update `.spec/` and `.agents/` when architecture or process changes occur.

## Operations Team
- **Deployment**: Execute scheduled syncs via `.github/workflows/daily-r2-sync.yml`; monitor Qdrant health and R2 backups.
- **Observability**: Track metrics (embedding latency, batch success rate, memory footprint) and escalate anomalies.
- **Security**: Rotate credentials, audit GitHub Actions secrets, and enforce `.env` isolation.

## Subagent Behaviors
- **General Agent**: Used for multi-step research, code search, and complex task orchestration.
- **Spec-Driven**: All implementation follows `.spec/` as authoritative contract; deviations require spec update first.
- **Test-First**: Red → green → refactor loops only; no code commits on failing tests.

## Escalation Path
1. **Blockers**: Surface in `.tasks/<task>/PROGRESS.md` with links to relevant spec sections.
2. **High-Impact Changes**: Schema updates, security patches, major refactors (>200 LOC) require explicit approval gate.
3. **Production Incidents**: Trigger runbook in `.agents/20-workflows/` and update `PROGRESS.md` with remediation steps.
