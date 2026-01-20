"""Pydantic models for API request/response schemas."""

from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field


# Request Models
class AnalyzeRequest(BaseModel):
    """Request model for analysis endpoint."""

    upload_id: str = Field(..., description="ID of the uploaded file")
    target: Optional[str] = Field(None, description="Target column for regression")
    context: Optional[str] = Field(None, description="Business context hint for LLM")
    skip_llm: bool = Field(False, description="Skip LLM-powered analysis")


# Response Models
class UploadResponse(BaseModel):
    """Response model for file upload."""

    upload_id: str = Field(..., description="Unique ID for the upload")
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., description="File size in bytes")
    status: str = Field("uploaded", description="Upload status")


class JobResponse(BaseModel):
    """Response model for analysis job."""

    job_id: str = Field(..., description="Unique ID for the analysis job")
    status: str = Field(..., description="Job status")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service health status")
    llm_available: bool = Field(..., description="Whether LLM is available")
    providers: List[str] = Field(default_factory=list, description="Available LLM providers")


class ProviderInfo(BaseModel):
    """Information about an LLM provider."""

    name: str
    available: bool
    models: List[str] = Field(default_factory=list)


class ProvidersResponse(BaseModel):
    """Response model for providers list."""

    providers: List[ProviderInfo]


class ColumnStatistics(BaseModel):
    """Statistics for a single column."""

    name: str
    dtype: str
    count: int
    null_count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[Any] = None
    max: Optional[Any] = None
    unique_count: Optional[int] = None


class RegressionResult(BaseModel):
    """Result from regression analysis."""

    target: str
    features: List[str]
    r_squared: float
    coefficients: Dict[str, float]
    p_values: Dict[str, float]
    feature_importance: Dict[str, float]


class AnalysisResultResponse(BaseModel):
    """Response model for analysis results."""

    status: str = Field(..., description="Analysis status")
    file_path: str = Field(..., description="Original file path")
    rows: int = Field(..., description="Number of rows")
    columns: int = Field(..., description="Number of columns")
    transposed: bool = Field(False, description="Whether data was transposed")
    transpose_decision: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    column_analysis: Optional[Dict[str, Any]] = None
    statistics: Optional[Dict[str, Any]] = None
    charts: List[str] = Field(default_factory=list, description="List of chart filenames")
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")


class ErrorResponse(BaseModel):
    """Response model for errors."""

    detail: str = Field(..., description="Error message")
