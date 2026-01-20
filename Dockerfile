# Build stage - build the wheel
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir hatchling build

# Copy source files needed for build
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Build wheel
RUN python -m build --wheel

# Runtime stage - minimal image for running the app
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies (curl for health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheel from build stage
COPY --from=builder /app/dist/*.whl /tmp/

# Install the package with api extras and all provider extras
RUN pip install --no-cache-dir /tmp/*.whl[api,all-providers] && rm /tmp/*.whl

# Create non-root user
RUN useradd -m -u 1000 appuser

# Create data directories (owned by appuser)
RUN mkdir -p /app/data /app/output && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default environment variables
ENV LOG_LEVEL=INFO
ENV PORT=8000

# Entry point
CMD ["uvicorn", "regression_analyzer.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
