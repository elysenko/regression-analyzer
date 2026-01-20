# tests/stats/test_feature_importance.py

import pytest
import polars as pl
import numpy as np

from regression_analyzer.stats import calculate_feature_importance
from regression_analyzer.stats.models import FeatureImportanceResult


class TestFeatureImportance:
    """Tests for feature importance analysis."""

    def test_feature_importance_ranking(self):
        """Test that important features rank higher."""
        np.random.seed(42)
        n = 200

        # y strongly depends on x1, weakly on x2, not on x3
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)  # noise
        y = 5 * x1 + 0.5 * x2 + np.random.randn(n) * 0.1

        df = pl.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})

        result = calculate_feature_importance(df, target="y", features=["x1", "x2", "x3"])

        # x1 should be most important (rank 1)
        assert result.features[0].feature == "x1"
        assert result.features[0].rank == 1
        assert result.features[0].importance > result.features[1].importance

    def test_feature_importance_method(self):
        """Test that permutation importance is used."""
        np.random.seed(42)
        n = 100

        x = np.random.randn(n)
        y = 2 * x + np.random.randn(n) * 0.1

        df = pl.DataFrame({"x": x, "y": y})

        result = calculate_feature_importance(df, target="y", features=["x"])

        assert result.method == "permutation_importance"

    def test_feature_importance_std(self):
        """Test that importance standard deviation is computed."""
        np.random.seed(42)
        n = 100

        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 3 * x1 + x2 + np.random.randn(n) * 0.1

        df = pl.DataFrame({"x1": x1, "x2": x2, "y": y})

        result = calculate_feature_importance(df, target="y", features=["x1", "x2"])

        for f in result.features:
            assert f.importance_std >= 0

    def test_feature_importance_model_score(self):
        """Test that model R² is returned."""
        np.random.seed(42)
        n = 100

        x = np.random.randn(n)
        y = 2 * x + np.random.randn(n) * 0.1

        df = pl.DataFrame({"x": x, "y": y})

        result = calculate_feature_importance(df, target="y", features=["x"])

        assert 0 <= result.model_r_squared <= 1
        assert result.model_r_squared > 0.8  # Should be high for this data

    def test_feature_importance_interpretation(self):
        """Test that interpretation is generated."""
        np.random.seed(42)
        n = 100

        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 2 * x1 + np.random.randn(n) * 0.1

        df = pl.DataFrame({"x1": x1, "x2": x2, "y": y})

        result = calculate_feature_importance(df, target="y", features=["x1", "x2"])

        assert result.interpretation
        assert "Random Forest" in result.interpretation
        assert "causation" in result.interpretation.lower()

    def test_feature_importance_insufficient_data(self):
        """Test error handling for insufficient data."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_feature_importance(df, target="y", features=["x"])

    def test_feature_importance_multiple_features(self):
        """Test with multiple features."""
        np.random.seed(42)
        n = 100

        features_data = {f"x{i}": np.random.randn(n) for i in range(5)}
        # y depends primarily on x0 and x1
        features_data["y"] = 3 * features_data["x0"] + 2 * features_data["x1"] + np.random.randn(n) * 0.1

        df = pl.DataFrame(features_data)

        result = calculate_feature_importance(
            df, target="y",
            features=["x0", "x1", "x2", "x3", "x4"]
        )

        assert len(result.features) == 5

        # All should have ranks 1-5
        ranks = {f.rank for f in result.features}
        assert ranks == {1, 2, 3, 4, 5}

        # x0 or x1 should be at the top
        top_features = {result.features[0].feature, result.features[1].feature}
        assert "x0" in top_features or "x1" in top_features
