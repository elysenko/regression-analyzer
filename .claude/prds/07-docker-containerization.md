---
name: 07-docker-containerization
description: Containerize regression-analyzer for deployment
status: backlog
created: 2026-01-20T20:32:34Z
updated: 2026-01-20T20:32:34Z
---

# PRD: Docker Containerization

## Overview

Package the regression-analyzer FastAPI application as a Docker container for consistent deployment across environments.

## Problem Statement

The application needs to be:
- Deployable to Kubernetes clusters
- Consistent across dev/staging/production
- Isolated with all dependencies included
- Efficient in resource usage

## Requirements

### Functional Requirements

1. **Dockerfile**
   - Multi-stage build for smaller image size
   - Python 3.11+ base image
   - Include all dependencies from pyproject.toml
   - Configure uvicorn as entrypoint

2. **Build Configuration**
   - Support build arguments for version tagging
   - Health check instruction for container orchestration
   - Non-root user for security

3. **Environment Configuration**
   - Environment variables for API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY)
   - Configurable port (default 8000)
   - Log level configuration

4. **Volume Mounts**
   - /data for uploaded files
   - /output for generated charts

### Non-Functional Requirements

- Image size < 500MB
- Startup time < 10 seconds
- Memory limit: 512MB base, 2GB max
- CPU limit: 0.5 base, 2 max

## Technical Approach

### Dockerfile

```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app
COPY pyproject.toml .
RUN pip install build && python -m build --wheel

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime deps only
COPY --from=builder /app/dist/*.whl .
RUN pip install --no-cache-dir *.whl && rm *.whl

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Create data directories
RUN mkdir -p /app/data /app/output

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "regression_analyzer.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### .dockerignore

```
.git
.venv
.pytest_cache
__pycache__
*.pyc
tests/
docs/
.claude/
```

### docker-compose.yml (for local development)

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./output:/app/output
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LOG_LEVEL=INFO
```

## Success Criteria

- Docker image builds successfully
- Container starts and responds to health checks
- API endpoints accessible from host
- Image size under 500MB
- Passes security scan (no critical vulnerabilities)

## Dependencies

- PRD-06: FastAPI Web API must be implemented first

## Out of Scope

- Multi-architecture builds (ARM64)
- GPU support for ML acceleration
- Sidecar containers
