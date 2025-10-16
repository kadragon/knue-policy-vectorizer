---
id: AG-TEMPLATES-KNUE-001
version: 1.0.0
scope: global
status: active
supersedes: []
depends: [AG-WORKFLOW-KNUE-OPS-001]
last-updated: 2025-10-16
owner: team-admin
---

# Standard Templates

## Commit Message Format
```
[Structural|Behavioral](<scope>) <summary> [<task-slug>]

Example:
[Structural](sync_pipeline) Extract git watcher into separate module [refactor-src-folder]
[Behavioral](embedding) Add exponential backoff for OpenAI rate limits [PHASE-4]
```

## Pull Request Template
```markdown
## Summary
- <1-3 bullet points describing the change>

## Related Issue
Closes #123 or Links to .tasks/<task>/

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests run successfully
- [ ] Manual QA performed (if applicable)

## Checklist
- [ ] Code follows project style (black, isort, mypy)
- [ ] Tests pass locally (`uv run pytest -v`)
- [ ] Documentation updated (README, SPEC, AGENTS)
- [ ] No secrets or sensitive data committed
```

## Task Folder Structure
```
.tasks/<task-name>/
├── RESEARCH.md       # Hypotheses, metrics, evidence
├── SPEC-DELTA.md     # Acceptance criteria, scope, non-functional constraints
├── PLAN.md           # Outline, dependencies, rollback, validation
├── PROGRESS.md       # Status updates, blockers, completion log
└── README.md         # (optional) Task-specific documentation
```

## Documentation Template — Agent Policy File
```yaml
---
id: AG-<TYPE>-<DOMAIN>-<SEQ>
version: 1.0.0
scope: global | folder:<path>
status: active | deprecated
supersedes: [AG-OLD-ID]
depends: [AG-DEP-ID, SPEC-ID]
last-updated: YYYY-MM-DD
owner: team-name
---

# Title

## Summary
Brief description of policy, workflow, or role.

## Key Points
- Point 1
- Point 2

## References
- Link to related SPEC or AGENTS documents
```
