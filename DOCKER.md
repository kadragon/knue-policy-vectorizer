# Docker Deployment Guide

This guide covers how to run the KNUE Policy Vectorizer using Docker.

## Prerequisites

1. Docker and Docker Compose installed
2. Ollama running locally on the host machine with bge-m3 model
3. At least 2GB available memory

## Quick Start

### 1. Start Qdrant Database

```bash
docker-compose up -d qdrant
```

This will start the Qdrant vector database on ports 6333 (HTTP) and 6334 (gRPC).

### 2. Verify Ollama is Running

Ensure Ollama is running on your host machine:

```bash
curl http://localhost:11434/api/version
```

You should see a version response. If not, start Ollama and pull the bge-m3 model:

```bash
ollama serve
ollama pull bge-m3
```

### 3. Run One-time Sync

```bash
docker-compose run --rm indexer uv run python -m src.sync_pipeline sync
```

### 4. Run with Automatic Hourly Sync

```bash
docker-compose up -d
```

This will start both Qdrant and the indexer service, which will:
- Run an initial sync immediately
- Perform sync every hour automatically
- Restart automatically if it crashes

## Service Management

### View Logs

```bash
# View all logs
docker-compose logs -f

# View only indexer logs
docker-compose logs -f indexer

# View only Qdrant logs
docker-compose logs -f qdrant
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop only indexer (keep Qdrant running)
docker-compose stop indexer
```

### Health Check

```bash
# Check all services health
docker-compose run --rm indexer uv run python -m src.sync_pipeline health

# Check Qdrant directly
curl http://localhost:6333/collections
```

## Configuration Options

### Environment Variables

Create a `.env` file to override default settings:

```bash
# Copy the example file
cp .env.example .env

# Edit with your settings
vim .env
```

Key environment variables:

- `QDRANT_URL`: Qdrant server URL (default: http://localhost:6333)
- `OLLAMA_URL`: Ollama server URL (default: http://localhost:11434)
- `GIT_REPO_URL`: Git repository to sync (default: KNUE-Policy-Hub)
- `COLLECTION_NAME`: Qdrant collection name (default: knue_policies)
- `LOG_LEVEL`: Logging level (default: INFO)

### Docker Compose Variants

1. **docker-compose.yml**: Simple hourly sync with loop
2. **docker-compose.cron.yml**: Uses cron for scheduling (more robust)

To use the cron version:

```bash
docker-compose -f docker-compose.cron.yml up -d
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```bash
   # Verify Ollama is accessible from Docker
   docker run --rm --add-host=host.docker.internal:host-gateway alpine:latest sh -c "wget -qO- http://host.docker.internal:11434/api/version"
   ```

2. **Qdrant Connection Failed**
   ```bash
   # Check if Qdrant is running
   docker-compose ps qdrant
   
   # Check Qdrant logs
   docker-compose logs qdrant
   ```

3. **Permission Issues**
   ```bash
   # Ensure directories are writable
   mkdir -p repo_cache logs
   chmod 755 repo_cache logs
   ```

4. **Memory Issues**
   ```bash
   # Check Docker memory limits
   docker system info | grep Memory
   
   # Clean up old containers/images
   docker system prune -f
   ```

### Accessing Qdrant Dashboard

Visit http://localhost:6333/dashboard to access the Qdrant web interface and inspect your vectors.

### Manual Commands

```bash
# Run specific sync commands
docker-compose run --rm indexer uv run python -m src.sync_pipeline sync
docker-compose run --rm indexer uv run python -m src.sync_pipeline reindex
docker-compose run --rm indexer uv run python -m src.sync_pipeline health

# Debug container
docker-compose run --rm indexer bash
```

## Production Deployment

For production deployment:

1. Use `docker-compose.cron.yml` for better reliability
2. Set up log rotation:
   ```bash
   # Add to /etc/logrotate.d/knue-vectorizer
   /path/to/logs/*.log {
       daily
       rotate 7
       compress
       missingok
       notifempty
   }
   ```

3. Monitor services with health checks
4. Set up backup for Qdrant data:
   ```bash
   # Backup Qdrant volume
   docker run --rm -v knue-policy-vectorizer_qdrant_storage:/data -v $(pwd):/backup alpine tar czf /backup/qdrant-backup.tar.gz -C /data .
   ```

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

The indexer service:
1. Fetches markdown files from Git repository
2. Processes them through Ollama (on host) for embeddings
3. Stores vectors in Qdrant database
4. Runs on schedule to keep data synchronized