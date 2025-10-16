---
id: AG-OVERRIDES-KNUE-001
version: 1.0.0
scope: global
status: active
supersedes: []
depends: []
last-updated: 2025-10-16
owner: team-admin
---

# Overrides & Exceptions

## Current Exceptions
- None active as of 2025-10-16.

## Historical Patches (Deprecated)
- To be archived in `.agents/_archive/` once superseded.

## Emergency Overrides
If a critical production incident requires bypassing standard procedures:
1. Document the incident in `.tasks/<incident>/PROGRESS.md`.
2. Record the override with timestamp and rationale in this file.
3. Schedule a post-incident review to update `.agents/` policies.

## Template for Recording Overrides
```markdown
### Override: <descriptive name>
- **Date**: YYYY-MM-DD HH:MM UTC
- **Reason**: <brief incident description>
- **Scope**: <affected services/components>
- **Duration**: <from HH:MM to HH:MM>
- **Remediation**: <steps taken>
- **Review**: [Link to post-incident review]
- **Status**: [resolved | ongoing]
```
