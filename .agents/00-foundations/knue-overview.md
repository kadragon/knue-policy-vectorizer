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
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Host Machine  │    │  Docker Network │    │  Docker Network │
│                 │    │                 │    │                 │
│  Ollama:11434   │◄───┤  Indexer        │◄───┤  Qdrant:6333    │
│  bge-m3 model   │    │  (Python App)   │    │  (Vector DB)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Git Repository │
                    │  KNUE-Policy-Hub│
                    └─────────────────┘
```

## Technology Stack
- Python 3.11+, managed via `uv`.
- Qdrant (1024-dimensional vectors) for vector persistence.
- Ollama `bge-m3` model for embeddings.
- Docker & Docker Compose for container orchestration.
- `structlog` for structured logging across services.

## Repository Layout
```
knue-policy-vectorizer/
├── src/                        # Core application modules
├── tests/                      # pytest suite (unit/integration/slow markers)
├── scripts/                    # Integration utilities (e.g., test_full_sync_pipeline.py)
├── config/                     # Configuration templates
├── test_data/                  # Representative fixtures
├── logs/                       # Runtime logs (gitignored)
├── repo_cache/                 # Local clone of watched repo
├── docker-compose*.yml         # Deployment topologies
├── Dockerfile                  # Indexer container image
├── README.md / DOCKER.md       # External documentation
└── TODO.md                     # Development history & backlog
```

## Future Roadmap
- Scalability: distributed processing, Redis caching, multi-repository ingestion.
- Monitoring: Prometheus metrics, alerting integrations, performance dashboards.
- Feature expansion: PDF/DOCX parsing, multilingual support, advanced chunking strategies.
- Quality initiatives: property-based testing, load testing, resilience validation.
- Security enhancements: Qdrant authentication, strict input validation, rate limiting.
