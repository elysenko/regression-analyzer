---
name: 06-fastapi-web-api
status: completed
created: 2026-01-20T20:35:47Z
progress: 100%
prd: .claude/prds/06-fastapi-web-api.md
github: https://github.com/elysenko/regression-analyzer/issues/21
---

# Epic: FastAPI Web API

## Overview

Wrap the existing regression-analyzer CLI pipeline in FastAPI REST endpoints. The existing `AnalysisPipeline` class already supports async operations and JSON serialization via `to_dict()`, making it straightforward to expose as a REST API.

## Architecture Decisions

- **Leverage existing pipeline**: The `AnalysisPipeline` class already handles all analysis logic with async methods. The API layer will be a thin wrapper.
- **Simple file storage**: Use local filesystem with UUID-based directories for uploaded files and results. No database needed for MVP.
- **Sync execution for simplicity**: Given the existing async pipeline, run analysis synchronously within endpoint handlers (analysis typically completes in seconds for reasonable file sizes).
- **Single module structure**: Keep API code minimal - one main.py with route handlers, one models.py for Pydantic schemas.

## Technical Approach

### Backend Services

The API wraps the existing pipeline:

1. **Upload Handler**: Save uploaded file to temp storage, return upload_id (UUID)
2. **Analyze Handler**: Load file, run `AnalysisPipeline`, store results, return job_id
3. **Results Handler**: Retrieve stored analysis results by job_id
4. **Charts Handler**: Serve generated chart images from disk
5. **Health Handler**: Return status and LLM provider availability

### Module Structure

```
src/regression_analyzer/api/
├── __init__.py
├── main.py           # FastAPI app with all routes
├── models.py         # Pydantic request/response schemas
└── storage.py        # Simple file-based storage for uploads/results
```

### Dependencies

- FastAPI >= 0.109.0
- python-multipart (file uploads)
- uvicorn (ASGI server)

## Implementation Strategy

1. Create Pydantic models for all request/response schemas
2. Implement simple file storage abstraction
3. Create FastAPI app with all endpoints in a single file
4. Add tests for each endpoint

## Task Breakdown Preview

- [ ] Task 1: Create Pydantic models and storage module
- [ ] Task 2: Create FastAPI app with upload and health endpoints
- [ ] Task 3: Add analyze and results endpoints
- [ ] Task 4: Add charts endpoint and CORS configuration
- [ ] Task 5: Add API tests

## Dependencies

- Existing `AnalysisPipeline` class (already async-ready)
- Existing `AnalysisResult.to_dict()` for JSON serialization
- FastAPI, python-multipart, uvicorn packages

## Success Criteria (Technical)

- All 5 endpoint groups functional: upload, analyze, results, charts, health
- File upload accepts CSV and Excel up to 50MB
- Analysis results returned as JSON matching existing `to_dict()` format
- Charts served as PNG images
- OpenAPI docs auto-generated at /docs
- Basic integration tests pass

## Estimated Effort

- Overall: 1-2 days implementation
- Low risk: Wrapping existing tested pipeline functionality
- Critical path: Storage module and analyze endpoint

## Tasks Created

- [ ] #22 - Create Pydantic models and storage module (parallel: true)
- [ ] #23 - Create FastAPI app with upload and health endpoints (parallel: false)
- [ ] #24 - Add analyze and results endpoints (parallel: false)
- [ ] #25 - Add charts endpoint and finalize API (parallel: false)
- [ ] #26 - Add API integration tests (parallel: false)

Total tasks: 5
Parallel tasks: 1
Sequential tasks: 4
Estimated total effort: 10-14 hours
