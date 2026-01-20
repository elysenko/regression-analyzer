---
name: 07-docker-containerization
status: backlog
created: 2026-01-20T20:45:03Z
updated: 2026-01-20T20:47:26Z
progress: 0%
prd: .claude/prds/07-docker-containerization.md
github: https://github.com/elysenko/regression-analyzer/issues/27
---

# Epic: Docker Containerization

## Overview

Containerize the regression-analyzer FastAPI application using Docker with multi-stage builds for efficient deployment. This creates production-ready container images with proper security (non-root user), health checks, and local development support via docker-compose.

## Architecture Decisions

- **Multi-stage build**: Separate build and runtime stages to minimize final image size (<500MB target)
- **Python 3.11-slim base**: Balances compatibility with image size
- **Hatchling build**: Use existing pyproject.toml build system rather than pip install from source
- **Non-root execution**: Create dedicated appuser (UID 1000) for security
- **Curl-based health check**: Simple HTTP health check against /api/v1/health endpoint

## Technical Approach

### Container Build

1. **Build stage**: Install build dependencies, build wheel from pyproject.toml
2. **Runtime stage**: Copy only the built wheel, install it, configure non-root user
3. **Entry point**: uvicorn with configurable host/port via environment variables

### Configuration

- Environment variables for API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY)
- LOG_LEVEL for configurable logging
- PORT for flexible port binding (default 8000)

### Volume Strategy

- `/app/data`: Input files (uploaded CSVs/Excel)
- `/app/output`: Generated charts and reports

## Implementation Strategy

1. Create Dockerfile with multi-stage build
2. Create .dockerignore for build context optimization
3. Create docker-compose.yml for local development
4. Test build and runtime behavior
5. Validate image size and startup time

## Task Breakdown Preview

- [ ] Task 1: Create multi-stage Dockerfile with security best practices
- [ ] Task 2: Create .dockerignore for optimized build context
- [ ] Task 3: Create docker-compose.yml for local development
- [ ] Task 4: Build and test the container

## Dependencies

- FastAPI API implementation (PRD-06) - the api/ module must exist
- pyproject.toml with api extras defined

## Success Criteria (Technical)

- Docker image builds without errors
- Image size < 500MB
- Container starts in < 10 seconds
- Health check endpoint responds correctly
- API endpoints accessible from host machine
- Non-root user execution verified

## Estimated Effort

- Overall: 2-3 hours
- Critical path: Dockerfile creation and testing
- Resource: Single developer

## Tasks Created

- [ ] #28 - Create multi-stage Dockerfile (parallel: true)
- [ ] #29 - Create .dockerignore file (parallel: true)
- [ ] #30 - Create docker-compose.yml for local development (parallel: false)
- [ ] #31 - Build and test Docker container (parallel: false)

Total tasks: 4
Parallel tasks: 2
Sequential tasks: 2
Estimated total effort: 2 hours
