"""FastAPI application with all REST endpoints."""

import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .models import (
    AnalysisResultResponse,
    AnalyzeRequest,
    ErrorResponse,
    HealthResponse,
    JobResponse,
    ProviderInfo,
    ProvidersResponse,
    UploadResponse,
)
from .storage import Storage

# Initialize app
app = FastAPI(
    title="Regression Analyzer API",
    description="REST API for automated tabular data analysis with LLM-powered insights",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize storage
storage = Storage()

# Allowed file extensions and max size
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def _check_llm_provider(name: str) -> tuple[bool, List[str]]:
    """Check if an LLM provider is available.

    Args:
        name: Provider name (openai, anthropic, google, groq)

    Returns:
        Tuple of (available, list of models)
    """
    import os

    if name == "openai":
        if os.getenv("OPENAI_API_KEY"):
            return True, ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
    elif name == "anthropic":
        if os.getenv("ANTHROPIC_API_KEY"):
            return True, ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]
    elif name == "google":
        if os.getenv("GOOGLE_API_KEY"):
            return True, ["gemini-1.5-pro", "gemini-1.5-flash"]
    elif name == "groq":
        if os.getenv("GROQ_API_KEY"):
            return True, ["llama-3.1-70b-versatile", "mixtral-8x7b-32768"]

    return False, []


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check service health and LLM availability."""
    providers = ["openai", "anthropic", "google", "groq"]
    available_providers = []

    for provider in providers:
        available, _ = _check_llm_provider(provider)
        if available:
            available_providers.append(provider)

    return HealthResponse(
        status="healthy",
        llm_available=len(available_providers) > 0,
        providers=available_providers,
    )


@app.get("/api/v1/providers", response_model=ProvidersResponse, tags=["Health"])
async def list_providers():
    """List available LLM providers and their models."""
    providers = ["openai", "anthropic", "google", "groq"]
    result = []

    for provider in providers:
        available, models = _check_llm_provider(provider)
        result.append(ProviderInfo(name=provider, available=available, models=models))

    return ProvidersResponse(providers=result)


@app.post(
    "/api/v1/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["Upload"],
)
async def upload_file(file: UploadFile = File(...)):
    """Upload a CSV or Excel file for analysis.

    - **file**: CSV (.csv) or Excel (.xlsx, .xls) file
    - Max size: 50MB
    """
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read and validate file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB",
        )

    # Save file
    upload_id = storage.save_upload(content, file.filename)

    return UploadResponse(
        upload_id=upload_id,
        filename=file.filename,
        size=len(content),
        status="uploaded",
    )


@app.post(
    "/api/v1/analyze",
    response_model=JobResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Analysis"],
)
async def analyze(request: AnalyzeRequest):
    """Run analysis on an uploaded file.

    - **upload_id**: ID from upload endpoint
    - **target**: Optional target column for regression
    - **context**: Optional business context hint for LLM
    - **skip_llm**: Skip LLM-powered analysis (faster, but less intelligent)
    """
    # Validate upload exists
    upload_path = storage.get_upload_path(request.upload_id)
    if not upload_path:
        raise HTTPException(status_code=404, detail="Upload not found")

    # Generate job ID
    job_id = str(uuid.uuid4())

    try:
        # Import pipeline here to avoid startup delay
        from ..pipeline import AnalysisPipeline

        # Create pipeline with job output directory
        job_dir = storage.get_job_dir(job_id)
        pipeline = AnalysisPipeline(
            output_dir=str(job_dir),
            use_llm=not request.skip_llm,
        )

        # Run analysis
        await pipeline.load_data(str(upload_path))

        if not request.skip_llm:
            await pipeline.check_transpose()
            await pipeline.identify_context(request.context)
            await pipeline.analyze_columns()

        await pipeline.run_statistics(target=request.target)
        await pipeline.generate_charts()

        # Get and save result
        result = pipeline.get_result()
        result_dict = result.to_dict()
        result_dict["status"] = "completed"

        # Convert chart paths to just filenames
        if result_dict.get("charts"):
            result_dict["charts"] = [Path(c).name for c in result_dict["charts"]]

        storage.save_result(job_id, result_dict)

        return JobResponse(job_id=job_id, status="completed")

    except Exception as e:
        # Save error result
        storage.save_result(
            job_id,
            {
                "status": "failed",
                "error": str(e),
                "file_path": str(upload_path),
            },
        )
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get(
    "/api/v1/results/{job_id}",
    response_model=AnalysisResultResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Results"],
)
async def get_results(job_id: str):
    """Get analysis results by job ID.

    - **job_id**: ID from analyze endpoint
    """
    result = storage.get_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # If job failed, return the error
    if result.get("status") == "failed":
        raise HTTPException(status_code=500, detail=result.get("error", "Analysis failed"))

    return AnalysisResultResponse(**result)


@app.get(
    "/api/v1/charts/{job_id}/{chart_name}",
    responses={404: {"model": ErrorResponse}},
    tags=["Charts"],
)
async def get_chart(job_id: str, chart_name: str):
    """Get a generated chart image.

    - **job_id**: ID from analyze endpoint
    - **chart_name**: Chart filename (e.g., regression_plot.png)
    """
    chart_path = storage.get_chart_path(job_id, chart_name)
    if chart_path is None:
        raise HTTPException(status_code=404, detail="Chart not found")

    return FileResponse(
        path=str(chart_path),
        media_type="image/png",
        filename=chart_name,
    )


@app.get("/api/v1/charts/{job_id}", tags=["Charts"])
async def list_charts(job_id: str):
    """List available charts for a job.

    - **job_id**: ID from analyze endpoint
    """
    result = storage.get_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found")

    charts = storage.list_charts(job_id)
    return {"job_id": job_id, "charts": charts}
