# tests/stats/test_stats_engine.py

import pytest
import polars as pl
import numpy as np

from regression_analyzer.stats import StatsEngine
from regression_analyzer.stats.models import StatisticsReport


class TestStatsEngine:
    """Tests for StatsEngine orchestrator."""

    def test_stats_engine_full_analysis(self):
        """Test full analysis with all components."""
        np.random.seed(42)
        n = 100

        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 2 * x1 + 3 * x2 + np.random.randn(n) * 0.1

        df = pl.DataFrame({"x1": x1, "x2": x2, "y": y})

        engine = StatsEngine()
        report = engine.analyze(df, target="y", features=["x1", "x2"])

        # All components should be present
        assert len(report.minmax) > 0
        assert report.linear_regression is not None
        assert report.feature_importance is not None

    def test_stats_engine_auto_detect_target(self):
        """Test auto-detection of target column."""
        np.random.seed(42)
        n = 100

        df = pl.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "target": np.random.randn(n)
        })

        engine = StatsEngine()
        report = engine.analyze(df)

        # Should auto-select last column as target
        assert report.linear_regression.target == "target"
        assert "Auto-selected target" in " ".join(report.warnings)

    def test_stats_engine_auto_detect_features(self):
        """Test auto-detection of feature columns."""
        np.random.seed(42)
        n = 100

        df = pl.DataFrame({
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            "c": np.random.randn(n)
        })

        engine = StatsEngine()
        report = engine.analyze(df, target="c")

        # Should use a and b as features
        assert set(report.linear_regression.features) == {"a", "b"}
        assert "Auto-selected" in " ".join(report.warnings)

    def test_stats_engine_skip_regression(self):
        """Test that regression can be skipped."""
        np.random.seed(42)
        n = 100

        df = pl.DataFrame({
            "x": np.random.randn(n),
            "y": np.random.randn(n)
        })

        engine = StatsEngine()
        report = engine.analyze(df, target="y", features=["x"], run_regression=False)

        assert report.linear_regression is None
        assert report.feature_importance is not None

    def test_stats_engine_skip_importance(self):
        """Test that importance can be skipped."""
        np.random.seed(42)
        n = 100

        df = pl.DataFrame({
            "x": np.random.randn(n),
            "y": np.random.randn(n)
        })

        engine = StatsEngine()
        report = engine.analyze(df, target="y", features=["x"], run_importance=False)

        assert report.linear_regression is not None
        assert report.feature_importance is None

    def test_stats_engine_insufficient_columns(self):
        """Test handling of insufficient numeric columns."""
        df = pl.DataFrame({
            "text1": ["a", "b", "c"],
            "text2": ["x", "y", "z"]
        })

        engine = StatsEngine()
        report = engine.analyze(df)

        assert len(report.minmax) == 0
        assert report.linear_regression is None
        assert report.feature_importance is None
        assert "Fewer than 2 numeric columns" in " ".join(report.warnings)

    def test_stats_engine_error_handling(self):
        """Test that errors are captured as warnings."""
        # Very small dataset that will fail for feature importance
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        })

        engine = StatsEngine()
        report = engine.analyze(df, target="y", features=["x"])

        # Regression might work, but importance needs 20+ samples
        # Either way, report should be returned with warnings
        assert isinstance(report, StatisticsReport)

    def test_stats_engine_minmax_always_runs(self):
        """Test that minmax analysis always runs."""
        np.random.seed(42)
        n = 50

        df = pl.DataFrame({
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            "c": np.random.randn(n)
        })

        engine = StatsEngine()
        report = engine.analyze(
            df, target="c", features=["a", "b"],
            run_regression=False, run_importance=False
        )

        # Minmax should still run
        assert len(report.minmax) == 3

    def test_stats_engine_report_structure(self):
        """Test that report has expected structure."""
        np.random.seed(42)
        n = 100

        df = pl.DataFrame({
            "x": np.random.randn(n),
            "y": np.random.randn(n)
        })

        engine = StatsEngine()
        report = engine.analyze(df)

        # Validate it's a proper StatisticsReport
        assert isinstance(report, StatisticsReport)
        assert isinstance(report.minmax, list)
        assert isinstance(report.warnings, list)
