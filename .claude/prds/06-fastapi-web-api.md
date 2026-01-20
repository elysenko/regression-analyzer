---
name: 06-fastapi-web-api
description: Wrap regression-analyzer CLI in FastAPI REST endpoints
status: backlog
created: 2026-01-20T20:32:34Z
updated: 2026-01-20T20:32:34Z
---

# PRD: FastAPI Web API

## Overview

Expose the regression-analyzer CLI functionality as a REST API using FastAPI. This enables web frontends, integrations, and remote access to the analysis pipeline.

## Problem Statement

The regression-analyzer is currently CLI-only, limiting its use to local command-line execution. Users need:
- Web interface integration for file uploads
- Remote API access for automation
- Programmatic access from other services

## Requirements

### Functional Requirements

1. **File Upload Endpoint**
   - Accept CSV and Excel files via multipart form upload
   - Support drag-and-drop from web clients
   - Validate file types and sizes (max 50MB)
   - Return upload ID for tracking

2. **Analysis Endpoint**
   - Trigger analysis on uploaded files
   - Support all CLI options: target, context, skip-llm
   - Return analysis results as JSON
   - Support async processing for large files

3. **Results Endpoint**
   - Retrieve analysis results by ID
   - Return statistics, charts (as base64 or URLs), warnings
   - Support polling for async job status

4. **Charts Endpoint**
   - Serve generated chart images
   - Support PNG format
   - Cache charts for repeated access

5. **Health Endpoint**
   - Health check for Kubernetes probes
   - Return LLM provider availability status

### Non-Functional Requirements

- Response time < 500ms for file upload
- Support concurrent analysis jobs
- Rate limiting: 10 requests/minute per IP
- CORS support for web frontends

## API Design

### Endpoints

```
POST   /api/v1/upload          - Upload file
POST   /api/v1/analyze         - Start analysis
GET    /api/v1/results/{id}    - Get results
GET    /api/v1/charts/{id}/{name}  - Get chart image
GET    /api/v1/health          - Health check
GET    /api/v1/providers       - List LLM providers
```

### Request/Response Examples

```python
# Upload
POST /api/v1/upload
Content-Type: multipart/form-data
file: <binary>

Response:
{
  "upload_id": "abc123",
  "filename": "data.csv",
  "size": 12345,
  "status": "uploaded"
}

# Analyze
POST /api/v1/analyze
{
  "upload_id": "abc123",
  "target": "revenue",
  "context": "SaaS metrics",
  "skip_llm": false
}

Response:
{
  "job_id": "xyz789",
  "status": "processing"
}

# Results
GET /api/v1/results/xyz789

Response:
{
  "status": "completed",
  "file_path": "data.csv",
  "rows": 100,
  "columns": 5,
  "statistics": {...},
  "charts": ["chart1.png", "chart2.png"],
  "warnings": []
}
```

## Technical Approach

### Module Structure
```
src/regression_analyzer/
├── api/
│   ├── __init__.py
│   ├── main.py         # FastAPI app
│   ├── routes/
│   │   ├── upload.py   # File upload endpoints
│   │   ├── analyze.py  # Analysis endpoints
│   │   ├── results.py  # Results retrieval
│   │   └── health.py   # Health checks
│   ├── models.py       # Pydantic request/response models
│   ├── storage.py      # File storage abstraction
│   └── jobs.py         # Background job management
```

### Dependencies
- FastAPI >= 0.109.0
- python-multipart (file uploads)
- uvicorn (ASGI server)

## Success Criteria

- All 5 endpoint groups implemented and tested
- Integration tests with actual file uploads
- API documentation auto-generated via OpenAPI
- Concurrent request handling verified

## Out of Scope

- Authentication/authorization (future PRD)
- WebSocket for real-time progress (future PRD)
- Multi-tenant support (future PRD)
