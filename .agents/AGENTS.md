---
id: AG-LOADER-KNUE-000
version: 1.0.0
scope: global
status: active
supersedes: []
depends: []
last-updated: 2025-10-15
owner: team-admin
---

# KNUE Policy Vectorizer — Agents Loader

> Load modular policies for the repository. Follow the declared order; higher numeric prefixes override earlier folders when conflicts arise.

## Load Order
1. `.agents/00-foundations/**`
2. `.agents/10-policies/**`
3. `.agents/20-workflows/**`
4. `.agents/30-roles/**`
5. `.agents/40-templates/**`
6. `.agents/90-overrides/**`

## Notes
- All files must include the required metadata front matter.
- Local folder-level `AGENTS.md` files may refine or override this order.
- Deprecated content should move to `.agents/_archive/` with updated status.
