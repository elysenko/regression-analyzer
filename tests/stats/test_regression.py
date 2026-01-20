# tests/stats/test_regression.py

import pytest
import polars as pl
import numpy as np

from regression_analyzer.stats import run_linear_regression
from regression_analyzer.stats.models import LinearRegressionResult


class TestLinearRegression:
    """Tests for linear regression analysis."""

    def test_linear_regression_simple(self):
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
        assert abs(coef_dict["x1"] - 2.0) < 0.3
        assert abs(coef_dict["x2"] - 3.0) < 0.3

    def test_linear_regression_significance(self):
        """Test that p-values correctly identify significance."""
        np.random.seed(42)
        n = 100

        # Strong predictor and noise predictor
        x_strong = np.random.randn(n)
        x_noise = np.random.randn(n)
        y = 5 * x_strong + np.random.randn(n) * 0.5

        df = pl.DataFrame({
            "x_strong": x_strong,
            "x_noise": x_noise,
            "y": y
        })

        result = run_linear_regression(df, target="y", features=["x_strong", "x_noise"])

        # x_strong should be significant
        strong_coef = next(c for c in result.coefficients if c.feature == "x_strong")
        assert strong_coef.is_significant
        assert strong_coef.p_value is not None
        assert strong_coef.p_value < 0.05

    def test_linear_regression_metrics(self):
        """Test that all metrics are computed correctly."""
        np.random.seed(42)
        n = 100

        x = np.random.randn(n)
        y = 2 * x + np.random.randn(n) * 0.1

        df = pl.DataFrame({"x": x, "y": y})

        result = run_linear_regression(df, target="y", features=["x"])

        assert 0 <= result.r_squared <= 1
        assert 0 <= result.adjusted_r_squared <= 1
        assert result.rmse > 0
        assert result.n_samples > 0
        assert result.intercept is not None

    def test_linear_regression_interpretation(self):
        """Test that interpretation is generated."""
        np.random.seed(42)
        n = 50

        x = np.random.randn(n)
        y = 2 * x + np.random.randn(n) * 0.1

        df = pl.DataFrame({"x": x, "y": y})

        result = run_linear_regression(df, target="y", features=["x"])

        assert result.interpretation
        assert "R²" in result.interpretation or "fit" in result.interpretation.lower()

    def test_linear_regression_insufficient_data(self):
        """Test error handling for insufficient data."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0]
        })

        with pytest.raises(ValueError, match="Insufficient data"):
            run_linear_regression(df, target="y", features=["x"])

    def test_linear_regression_with_nulls(self):
        """Test that nulls are handled by dropping rows."""
        np.random.seed(42)
        n = 50

        x = list(np.random.randn(n))
        y = [2 * xi + np.random.randn() * 0.1 for xi in x]

        # Add some nulls
        x[5] = None
        y[10] = None

        df = pl.DataFrame({"x": x, "y": y})

        result = run_linear_regression(df, target="y", features=["x"])

        # Should still work with fewer samples
        assert result.n_samples < n
        assert result.r_squared > 0.9
