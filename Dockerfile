# ==============================================================================
# Base OS Stage: Common runtime tools, Prisma OpenSSL 3 support, and uv
# ==============================================================================
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_SYNC=1 \
    UV_NO_DEV=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/.venv/Lib/site-packages:/app"

# Install minimal OS libraries required for Prisma, PostgreSQL, Node/libatomic, curl for healthchecks
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libssl3 \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first to maximize Docker layer caching
COPY pyproject.toml uv.lock .python-version README.md ./

# ==============================================================================
# Target: core (Lean Production Image for FastAPI Web API & Core Services)
# Zero heavy ML/Torch dependencies: fast cold starts, minimal memory footprint (<250MB)
# ==============================================================================
FROM base AS core

# Install ONLY core project dependencies (without [optional-dependencies] ml and without dev)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Ensure Linux Python compatibility for Prisma schema output path (../.venv/Lib -> .venv/lib/python3.11)
RUN mkdir -p /app/.venv/lib/python3.11/site-packages && \
    ln -sfn /app/.venv/lib/python3.11 /app/.venv/Lib

# Copy Prisma schema and generate typed Prisma Client Python
COPY prisma/ ./prisma/
ENV DATABASE_URL="postgresql://placeholder:placeholder@localhost:5432/placeholder"
RUN uv run --no-dev prisma generate --schema=prisma/schema.prisma

# Copy application source code
COPY . .

# Finalize core project installation
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ==============================================================================
# Target: mcp (Standalone Model Context Protocol Server)
# Inherits from lean core stage; exposes SSE on port 8001 with healthcheck
# ==============================================================================
FROM core AS mcp

EXPOSE 8001
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=5 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uv", "run", "python", "-m", "app.infrastructure.mcp.server", "--transport", "sse", "--host", "0.0.0.0", "--port", "8001"]

# ==============================================================================
# Target: worker (Background Task ARQ Worker with Heavy ML & Docling Support)
# Adds OpenCV & PyTorch OS libraries and installs --extra ml for document parsing
# ==============================================================================
FROM core AS worker

# Install OS libraries required for deep learning, Docling, and OpenCV image parsing
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install heavy ML dependencies (docling, torch, torchvision)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ml

CMD ["uv", "run", "arq", "app.core.queue.WorkerSettings"]
