# src/regression_analyzer/stats/__init__.py

from typing import List, Optional
import polars as pl

from .models import (
    MinMaxResult,
    LinearRegressionResult,
    FeatureImportanceResult,
    StatisticsReport,
)
from .minmax import analyze_minmax
from .regression import run_linear_regression
from .feature_importance import calculate_feature_importance


class StatsEngine:
    """Orchestrate statistical analysis."""

    def __init__(self):
        self.warnings: List[str] = []

    def analyze(
        self,
        df: pl.DataFrame,
        target: Optional[str] = None,
        features: Optional[List[str]] = None,
        run_regression: bool = True,
        run_importance: bool = True
    ) -> StatisticsReport:
        """Run full statistical analysis.

        Args:
            df: Input DataFrame
            target: Target column (auto-detect if None)
            features: Feature columns (auto-detect if None)
            run_regression: Whether to run linear regression
            run_importance: Whether to calculate feature importance

        Returns:
            StatisticsReport with all analysis results
        """
        self.warnings = []

        # Auto-detect numeric columns
        numeric_cols = [
            col for col in df.columns
            if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]

        if len(numeric_cols) < 2:
            self.warnings.append("Fewer than 2 numeric columns found")
            return StatisticsReport(
                minmax=[],
                warnings=self.warnings
            )

        # Auto-select target and features if not provided
        if target is None:
            target = numeric_cols[-1]  # Last numeric column
            self.warnings.append(f"Auto-selected target: {target}")

        if features is None:
            features = [c for c in numeric_cols if c != target]
            self.warnings.append(f"Auto-selected {len(features)} features")

        # Min/max analysis
        minmax_results = analyze_minmax(df, numeric_cols)

        # Linear regression
        regression_result = None
        if run_regression and len(features) > 0:
            try:
                regression_result = run_linear_regression(df, target, features)
            except Exception as e:
                self.warnings.append(f"Regression failed: {e}")

        # Feature importance
        importance_result = None
        if run_importance and len(features) > 0:
            try:
                importance_result = calculate_feature_importance(df, target, features)
            except Exception as e:
                self.warnings.append(f"Feature importance failed: {e}")

        return StatisticsReport(
            minmax=minmax_results,
            linear_regression=regression_result,
            feature_importance=importance_result,
            warnings=self.warnings
        )


__all__ = [
    "StatsEngine",
    "analyze_minmax",
    "run_linear_regression",
    "calculate_feature_importance",
    "MinMaxResult",
    "LinearRegressionResult",
    "FeatureImportanceResult",
    "StatisticsReport",
]
