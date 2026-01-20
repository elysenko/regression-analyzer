# src/regression_analyzer/stats/models.py

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class MinMaxResult(BaseModel):
    """Min/max analysis for a column."""

    column: str
    min_value: float
    max_value: float
    min_index: int
    max_index: int
    min_context: Dict[str, Any] = Field(
        description="Other column values at min row"
    )
    max_context: Dict[str, Any] = Field(
        description="Other column values at max row"
    )
    range: float
    mean: float
    std: float


class RegressionCoefficient(BaseModel):
    """Single coefficient from linear regression."""

    feature: str
    coefficient: float
    std_error: Optional[float] = None
    t_statistic: Optional[float] = None
    p_value: Optional[float] = None
    is_significant: bool = Field(
        description="Whether p < 0.05"
    )


class LinearRegressionResult(BaseModel):
    """Results from linear regression analysis."""

    target: str
    features: List[str]
    coefficients: List[RegressionCoefficient]
    intercept: float
    r_squared: float
    adjusted_r_squared: float
    rmse: float
    n_samples: int
    interpretation: str = Field(
        description="Human-readable interpretation"
    )


class FeatureImportance(BaseModel):
    """Single feature importance score."""

    feature: str
    importance: float
    importance_std: float = Field(
        description="Standard deviation from permutation"
    )
    rank: int


class FeatureImportanceResult(BaseModel):
    """Results from feature importance analysis."""

    target: str
    features: List[FeatureImportance]
    model_r_squared: float
    method: str = "permutation_importance"
    interpretation: str


class StatisticsReport(BaseModel):
    """Complete statistics report."""

    minmax: List[MinMaxResult]
    linear_regression: Optional[LinearRegressionResult] = None
    feature_importance: Optional[FeatureImportanceResult] = None
    warnings: List[str] = Field(default_factory=list)
