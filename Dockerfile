# Use official lightweight Python image with pre-installed uv
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Set Python and uv runtime behaviors
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Install OS libraries for Prisma binaries, OpenSSL 3, and document processing (Docling/OpenCV/PDF)
# Use BuildKit cache mounts to avoid re-downloading Debian packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libssl3 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first to maximize Docker layer caching
COPY pyproject.toml uv.lock .python-version README.md ./

# Install project dependencies without the root project
# Use BuildKit cache mount for uv wheel downloads
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy Prisma schema and generate typed Prisma Client Python
COPY prisma/ ./prisma/
# Provide dummy fallback DATABASE_URL for schema parsing during build phase
ENV DATABASE_URL="postgresql://placeholder:placeholder@localhost:5432/placeholder"
RUN uv run prisma generate --schema=prisma/schema.prisma

# Copy application source code
COPY . .

# Finalize project installation
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Expose Web API port
EXPOSE 8000

# Default entrypoint for Web API (can be overridden by worker service in docker-compose)
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
