# src/regression_analyzer/stats/regression.py

from typing import List
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
