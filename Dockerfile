# =============================================================================
# Multi-stage Dockerfile for SaaS Agent Team
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specs + minimal package skeleton for pip install
COPY pyproject.toml ./
RUN mkdir -p agents api core memory observability orchestration tests && \
    touch agents/__init__.py api/__init__.py core/__init__.py \
    memory/__init__.py observability/__init__.py orchestration/__init__.py \
    tests/__init__.py && \
    touch README.md

# Install ALL dependencies (prod + dev for testing in CI)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "."

# =============================================================================
# Runtime stage
# =============================================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Ensure scripts are executable
RUN chmod +x start_worker.sh 2>/dev/null || true

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Expose port (Railway injects $PORT at runtime)
EXPOSE 8000

# Default command — Railway overrides via railway.toml startCommand
# Using shell form so $PORT is resolved at runtime
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2
