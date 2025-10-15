---
id: AG-FOUND-KNUE-001
version: 1.0.0
scope: global
status: active
supersedes: []
depends: []
last-updated: 2025-10-15
owner: team-admin
---

# KNUE Policy Vectorizer — Overview

## Purpose
- Synchronize Korean policy documents from the KNUE Policy Hub Git repository into a Qdrant vector database to power semantic search and RAG workloads.
- Provide engineers with a shared reference describing the system's intent and boundaries.

## Architecture
```
┌────────────────────┐       ┌──────────────────────┐
│  GitHub Actions    │       │  KNUE Policy Hub Git │
│  (Daily Sync)      │       │  Repository          │
└─────────┬──────────┘       └──────────┬───────────┘
          │                               │
          ▼                               ▼
┌────────────────────┐       ┌──────────────────────┐
│  Sync Pipeline CLI │──────▶│  Markdown Processor  │
│  (uv run …)        │       └──────────┬───────────┘
└─────────┬──────────┘                  │ chunks
          │ embeddings                  ▼
          ▼                    ┌──────────────────────┐
┌────────────────────┐         │  OpenAI Embeddings   │
│  Cloudflare R2     │         │  (text-embedding-3)  │
│  Document Cache    │────────▶└──────────┬───────────┘
└─────────┬──────────┘                    │ vectors (1536d)
          │                                ▼
          ▼                    ┌──────────────────────┐
┌────────────────────┐         │  Qdrant Cloud        │
│  Observability     │◀────────│  (Managed Vector DB) │
└────────────────────┘         └──────────────────────┘
```

## Technology Stack
- Python 3.11+, managed via `uv`.
- OpenAI Embeddings (`text-embedding-3-*`, 1536-dimensional vectors) served via HTTPS API.
- Qdrant Cloud for fully managed vector persistence and search.
- Cloudflare R2 for immutable document backups and soft-delete retention.
- `structlog` for structured logging across services.

## Repository Layout
```
knue-policy-vectorizer/
├── .spec/                     # Canonical roadmap & acceptance criteria
├── .agents/                   # Repository policies, workflows, templates
├── src/                        # Core application modules
├── tests/                      # pytest suite (unit/integration/slow markers)
├── scripts/                    # Integration utilities (e.g., test_full_sync_pipeline.py)
├── config/                     # Configuration templates
├── test_data/                  # Representative fixtures
├── logs/                       # Runtime logs (gitignored)
├── repo_cache/                 # Local clone of watched repo
└── README.md                   # Operator and developer documentation
```

## Future Roadmap
- Scalability: distributed processing, Redis caching, multi-repository ingestion.
- Monitoring: Prometheus metrics, alerting integrations, performance dashboards.
- Feature expansion: PDF/DOCX parsing, multilingual support, advanced chunking strategies.
- Quality initiatives: property-based testing, load testing, resilience validation.
- Security enhancements: Qdrant authentication, strict input validation, rate limiting.
