---
name: stats-engine
id: PRD-03
description: Statistical analysis with linear regression and random forest feature importance
status: backlog
phase: mvp
priority: P0
complexity: medium
wave: 3

depends_on:
  - PRD-01  # Needs DataLoader
  - PRD-02  # Needs column analysis for target/feature selection

creates:
  - src/regression_analyzer/stats/__init__.py
  - src/regression_analyzer/stats/regression.py
  - src/regression_analyzer/stats/feature_importance.py
  - src/regression_analyzer/stats/minmax.py
  - src/regression_analyzer/stats/models.py
  - tests/stats/test_regression.py
  - tests/stats/test_feature_importance.py
  - tests/stats/test_minmax.py

modifies: []

database:
  creates: []
  modifies: []

test_command: pytest tests/stats/

blocks: [PRD-04, PRD-05]

references: [PRD-01, PRD-02]

created: 2026-01-20T19:06:34Z
updated: 2026-01-20T19:06:34Z
---

# PRD-03: Statistics Engine

## Overview

**Feature:** Statistical analysis with linear regression and random forest feature importance
**Priority:** P0 (Core analysis capability)
**Complexity:** Medium
**Dependencies:** PRD-01 (Data Loader), PRD-02 (LLM Analyzer)

---

## Problem Statement

After identifying relevant columns, the system needs to:
1. Find min/max values and their context
2. Run linear regression to find relationships
3. Use Random Forest for feature importance (causation hints)
4. Provide actionable insights about what drives key metrics

Research findings:
- Use `permutation_importance()` not tree's `.feature_importances_` (unbiased)
- sklearn 1.4+ has native Polars support via `set_config(transform_output="polars")`
- Linear regression provides interpretable coefficients
- Random Forest captures non-linear relationships

---

## Goals

1. Identify min/max values for each column with context
2. Run linear regression with coefficient interpretation
3. Calculate feature importance using permutation importance
4. Provide R² scores and model quality metrics
5. Return structured results for charting and reporting

---

## Non-Goals

- Deep learning models
- Time series specific analysis (ARIMA, etc.)
- Hyperparameter tuning
- Cross-validation (simple train/test split for MVP)
- Causal inference (correlation != causation disclaimer)

---

## Technical Design

### Architecture

```
stats/
├── __init__.py           # Public API exports
├── regression.py         # Linear regression analysis
├── feature_importance.py # Random Forest + permutation importance
├── minmax.py            # Min/max identification
└── models.py            # Result models
```

### Result Models

```python
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
```

### Min/Max Analysis

```python
# src/regression_analyzer/stats/minmax.py

from typing import List
import polars as pl

from .models import MinMaxResult


def analyze_minmax(
    df: pl.DataFrame,
    columns: List[str] | None = None
) -> List[MinMaxResult]:
    """Analyze min/max values for numeric columns.

    Args:
        df: Input DataFrame
        columns: Specific columns to analyze (None = all numeric)

    Returns:
        List of MinMaxResult for each column
    """
    results = []

    # Get numeric columns
    if columns is None:
        columns = [
            col for col in df.columns
            if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]
        ]

    for col in columns:
        if col not in df.columns:
            continue

        series = df[col]

        # Skip if all null
        if series.null_count() == series.len():
            continue

        min_val = series.min()
        max_val = series.max()

        # Find indices
        min_idx = series.arg_min()
        max_idx = series.arg_max()

        # Get context (other values at min/max rows)
        min_context = {
            c: df[c][min_idx]
            for c in df.columns if c != col
        }
        max_context = {
            c: df[c][max_idx]
            for c in df.columns if c != col
        }

        results.append(MinMaxResult(
            column=col,
            min_value=float(min_val),
            max_value=float(max_val),
            min_index=int(min_idx),
            max_index=int(max_idx),
            min_context=min_context,
            max_context=max_context,
            range=float(max_val - min_val),
            mean=float(series.mean()),
            std=float(series.std()),
        ))

    return results
```

### Linear Regression

```python
# src/regression_analyzer/stats/regression.py

from typing import List, Optional
import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats

from .models import LinearRegressionResult, RegressionCoefficient


def run_linear_regression(
    df: pl.DataFrame,
    target: str,
    features: List[str],
    test_size: float = 0.2
) -> LinearRegressionResult:
    """Run linear regression analysis.

    Args:
        df: Input DataFrame
        target: Target column name
        features: Feature column names
        test_size: Fraction for test split

    Returns:
        LinearRegressionResult with coefficients and metrics
    """
    # Prepare data
    df_clean = df.select([target] + features).drop_nulls()

    if df_clean.height < 10:
        raise ValueError(f"Insufficient data: {df_clean.height} rows after dropping nulls")

    X = df_clean.select(features).to_numpy()
    y = df_clean.select(target).to_numpy().flatten()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    n = len(y_train)
    p = len(features)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Calculate standard errors and p-values
    coefficients = _calculate_coefficient_stats(
        model, X_train, y_train, features
    )

    # Generate interpretation
    interpretation = _interpret_regression(
        target, coefficients, r2, rmse
    )

    return LinearRegressionResult(
        target=target,
        features=features,
        coefficients=coefficients,
        intercept=float(model.intercept_),
        r_squared=float(r2),
        adjusted_r_squared=float(adj_r2),
        rmse=float(rmse),
        n_samples=n,
        interpretation=interpretation
    )


def _calculate_coefficient_stats(
    model: LinearRegression,
    X: np.ndarray,
    y: np.ndarray,
    features: List[str]
) -> List[RegressionCoefficient]:
    """Calculate coefficient statistics including p-values."""
    n = len(y)
    p = len(features)

    # Predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = np.sum(residuals ** 2) / (n - p - 1)

    # Standard errors
    X_with_intercept = np.column_stack([np.ones(n), X])
    try:
        var_coef = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        std_errors = np.sqrt(np.diag(var_coef))[1:]  # Skip intercept
    except np.linalg.LinAlgError:
        std_errors = [None] * p

    coefficients = []
    for i, feature in enumerate(features):
        coef = float(model.coef_[i])
        std_err = float(std_errors[i]) if std_errors[i] is not None else None

        if std_err and std_err > 0:
            t_stat = coef / std_err
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - p - 1))
            is_sig = p_val < 0.05
        else:
            t_stat = None
            p_val = None
            is_sig = False

        coefficients.append(RegressionCoefficient(
            feature=feature,
            coefficient=coef,
            std_error=std_err,
            t_statistic=t_stat,
            p_value=p_val,
            is_significant=is_sig
        ))

    return coefficients


def _interpret_regression(
    target: str,
    coefficients: List[RegressionCoefficient],
    r2: float,
    rmse: float
) -> str:
    """Generate human-readable interpretation."""
    lines = []

    # R² interpretation
    if r2 > 0.7:
        lines.append(f"Strong model fit (R²={r2:.2f}): Features explain {r2*100:.0f}% of {target} variance.")
    elif r2 > 0.4:
        lines.append(f"Moderate model fit (R²={r2:.2f}): Features explain {r2*100:.0f}% of {target} variance.")
    else:
        lines.append(f"Weak model fit (R²={r2:.2f}): Features explain only {r2*100:.0f}% of {target} variance.")

    # Significant coefficients
    sig_coefs = [c for c in coefficients if c.is_significant]
    if sig_coefs:
        lines.append("Significant predictors:")
        for c in sorted(sig_coefs, key=lambda x: abs(x.coefficient), reverse=True):
            direction = "increases" if c.coefficient > 0 else "decreases"
            lines.append(f"  - {c.feature}: {target} {direction} by {abs(c.coefficient):.2f} per unit")

    return " ".join(lines)
```

### Feature Importance

```python
# src/regression_analyzer/stats/feature_importance.py

from typing import List
import polars as pl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from .models import FeatureImportanceResult, FeatureImportance


def calculate_feature_importance(
    df: pl.DataFrame,
    target: str,
    features: List[str],
    n_estimators: int = 100,
    n_repeats: int = 10,
    random_state: int = 42
) -> FeatureImportanceResult:
    """Calculate feature importance using Random Forest + permutation importance.

    Uses permutation_importance instead of tree's feature_importances_
    because it's model-agnostic and unbiased toward high-cardinality features.

    Args:
        df: Input DataFrame
        target: Target column name
        features: Feature column names
        n_estimators: Number of trees in forest
        n_repeats: Permutation importance repeats
        random_state: Random seed

    Returns:
        FeatureImportanceResult with ranked features
    """
    # Prepare data
    df_clean = df.select([target] + features).drop_nulls()

    if df_clean.height < 20:
        raise ValueError(f"Insufficient data: {df_clean.height} rows after dropping nulls")

    X = df_clean.select(features).to_numpy()
    y = df_clean.select(target).to_numpy().flatten()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Fit Random Forest
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Model score
    r2 = model.score(X_test, y_test)

    # Permutation importance (on test set)
    perm_importance = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    # Build results
    importance_scores = []
    for i, feature in enumerate(features):
        importance_scores.append({
            "feature": feature,
            "importance": float(perm_importance.importances_mean[i]),
            "importance_std": float(perm_importance.importances_std[i]),
        })

    # Sort by importance and add rank
    importance_scores.sort(key=lambda x: x["importance"], reverse=True)
    for rank, item in enumerate(importance_scores, 1):
        item["rank"] = rank

    feature_importances = [
        FeatureImportance(**item) for item in importance_scores
    ]

    # Generate interpretation
    interpretation = _interpret_importance(target, feature_importances, r2)

    return FeatureImportanceResult(
        target=target,
        features=feature_importances,
        model_r_squared=float(r2),
        method="permutation_importance",
        interpretation=interpretation
    )


def _interpret_importance(
    target: str,
    features: List[FeatureImportance],
    r2: float
) -> str:
    """Generate interpretation of feature importance."""
    lines = []

    lines.append(f"Random Forest model R²={r2:.2f}.")

    # Top features
    top_features = [f for f in features if f.importance > 0.01][:5]
    if top_features:
        lines.append(f"Top predictors of {target}:")
        for f in top_features:
            lines.append(f"  {f.rank}. {f.feature} (importance: {f.importance:.3f})")

    # Negligible features
    negligible = [f for f in features if f.importance < 0.001]
    if negligible:
        lines.append(f"{len(negligible)} features have negligible importance.")

    lines.append("Note: Importance indicates predictive power, not causation.")

    return " ".join(lines)
```

### Statistics Engine Orchestrator

```python
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
```

---

## Implementation Tasks

### Task 1: Result Models
- [ ] Create `models.py` with Pydantic result models
- [ ] Add MinMaxResult, LinearRegressionResult
- [ ] Add FeatureImportanceResult, StatisticsReport

### Task 2: Min/Max Analysis
- [ ] Implement `analyze_minmax()` function
- [ ] Include context from other columns
- [ ] Calculate basic statistics (mean, std, range)

### Task 3: Linear Regression
- [ ] Implement `run_linear_regression()` function
- [ ] Calculate coefficient p-values and significance
- [ ] Generate human-readable interpretation
- [ ] Handle edge cases (multicollinearity, insufficient data)

### Task 4: Feature Importance
- [ ] Implement `calculate_feature_importance()` with Random Forest
- [ ] Use `permutation_importance()` (not tree importances)
- [ ] Generate interpretation
- [ ] Rank features by importance

### Task 5: Stats Engine
- [ ] Create `StatsEngine` orchestrator class
- [ ] Auto-detect target and features if not provided
- [ ] Aggregate warnings and errors
- [ ] Return complete StatisticsReport

### Task 6: Testing
- [ ] Unit tests for minmax analysis
- [ ] Unit tests for regression with known data
- [ ] Unit tests for feature importance
- [ ] Integration test with full pipeline

---

## Testing Strategy

### Unit Tests
```python
# tests/stats/test_regression.py

import pytest
import polars as pl
import numpy as np

from regression_analyzer.stats import run_linear_regression

def test_linear_regression_simple():
    """Test linear regression with known relationship."""
    np.random.seed(42)
    n = 100

    # y = 2*x1 + 3*x2 + noise
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = 2 * x1 + 3 * x2 + np.random.randn(n) * 0.1

    df = pl.DataFrame({
        "x1": x1,
        "x2": x2,
        "y": y
    })

    result = run_linear_regression(df, target="y", features=["x1", "x2"])

    assert result.r_squared > 0.95
    assert len(result.coefficients) == 2

    # Check coefficients are close to true values
    coef_dict = {c.feature: c.coefficient for c in result.coefficients}
    assert abs(coef_dict["x1"] - 2.0) < 0.2
    assert abs(coef_dict["x2"] - 3.0) < 0.2

def test_feature_importance_ranking():
    """Test that important features rank higher."""
    np.random.seed(42)
    n = 200

    # y strongly depends on x1, weakly on x2, not on x3
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)  # noise
    y = 5 * x1 + 0.5 * x2 + np.random.randn(n) * 0.1

    df = pl.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})

    from regression_analyzer.stats import calculate_feature_importance
    result = calculate_feature_importance(df, target="y", features=["x1", "x2", "x3"])

    # x1 should be most important
    assert result.features[0].feature == "x1"
    assert result.features[0].importance > result.features[1].importance
```

---

## Acceptance Criteria

1. **Min/Max**: Correctly identifies min/max with row context
2. **Linear Regression**: R² > 0.95 for perfect linear data
3. **Coefficients**: Recovers true coefficients within 10% for known data
4. **P-values**: Correctly identifies significant predictors
5. **Feature Importance**: Top feature matches strongest predictor
6. **Interpretations**: Human-readable, accurate summaries

---

## Dependencies

- `polars>=0.20.0` - Data manipulation
- `scikit-learn>=1.4.0` - ML algorithms, permutation importance
- `scipy>=1.11.0` - Statistical tests
- `numpy>=1.24.0` - Numerical operations

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Multicollinearity | Warn when VIF > 5, note in interpretation |
| Insufficient data | Require minimum 20 samples, clear error message |
| All null columns | Skip with warning |
| Categorical features | Currently numeric only, note limitation |
