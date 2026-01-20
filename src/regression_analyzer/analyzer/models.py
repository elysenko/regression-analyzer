from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class BusinessContext(BaseModel):
    """Identified business context from data."""
    company_type: str = Field(description="Type of company (e.g., 'retail', 'SaaS', 'manufacturing')")
    industry: str = Field(description="Industry sector (e.g., 'technology', 'finance', 'healthcare')")
    likely_focus_areas: List[str] = Field(description="Business metrics they likely care about", default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in context identification")

class ColumnRelevance(BaseModel):
    """Relevance assessment for a single column."""
    column_name: str
    semantic_type: str = Field(description="What this column represents (e.g., 'revenue', 'date', 'customer_id')")
    relevance: Literal["high", "medium", "low"] = Field(description="Relevance to business objectives")
    is_target_candidate: bool = Field(description="Whether this could be a prediction target")
    is_feature_candidate: bool = Field(description="Whether this could be a useful feature")
    data_quality_notes: Optional[str] = Field(default=None, description="Any data quality concerns")

class ColumnAnalysis(BaseModel):
    """Full column relevance analysis."""
    columns: List[ColumnRelevance]
    recommended_target: Optional[str] = Field(default=None, description="Recommended target column for regression")
    recommended_features: List[str] = Field(default_factory=list, description="Recommended feature columns")

class HeaderAnalysis(BaseModel):
    """Analysis of table headers for transpose detection."""
    first_row_looks_like_headers: bool = Field(description="Whether first row appears to contain column headers")
    first_column_looks_like_headers: bool = Field(description="Whether first column appears to contain row headers")
    header_examples_row: List[str] = Field(description="Examples from first row")
    header_examples_col: List[str] = Field(description="Examples from first column")
    reasoning: str = Field(description="Explanation of header analysis")

class TransposeDecision(BaseModel):
    """Final transpose recommendation."""
    should_transpose: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
