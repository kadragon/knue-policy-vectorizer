FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv package manager
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# Create directories for data and logs
RUN mkdir -p repo_cache logs

# Set Python path and virtual environment
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

# Install the package in development mode
RUN uv pip install -e .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD uv run python -m src.sync_pipeline health || exit 1

# Default command
CMD ["uv", "run", "python", "-m", "src.sync_pipeline", "sync"]