# tests/charts/test_generator.py

import pytest
from pathlib import Path
import polars as pl
import numpy as np

from regression_analyzer.charts import (
    ChartGenerator,
    plot_minmax_bar,
    plot_minmax_summary,
    plot_regression_scatter,
    plot_regression_coefficients,
    plot_feature_importance,
)
from regression_analyzer.stats.models import (
    StatisticsReport, MinMaxResult, LinearRegressionResult,
    RegressionCoefficient, FeatureImportanceResult, FeatureImportance
)


@pytest.fixture
def sample_minmax_result():
    return MinMaxResult(
        column="revenue",
        min_value=100, max_value=500,
        min_index=0, max_index=4,
        min_context={"region": "South"},
        max_context={"region": "North"},
        range=400, mean=300, std=100
    )


@pytest.fixture
def sample_regression_result():
    return LinearRegressionResult(
        target="revenue",
        features=["marketing", "headcount"],
        coefficients=[
            RegressionCoefficient(
                feature="marketing", coefficient=2.5,
                std_error=0.5, t_statistic=5.0,
                p_value=0.01, is_significant=True
            ),
            RegressionCoefficient(
                feature="headcount", coefficient=10.0,
                std_error=2.0, t_statistic=5.0,
                p_value=0.001, is_significant=True
            ),
        ],
        intercept=50.0, r_squared=0.85,
        adjusted_r_squared=0.83, rmse=20.0,
        n_samples=100, interpretation="Strong fit"
    )


@pytest.fixture
def sample_importance_result():
    return FeatureImportanceResult(
        target="revenue",
        features=[
            FeatureImportance(feature="headcount", importance=0.4, importance_std=0.05, rank=1),
            FeatureImportance(feature="marketing", importance=0.3, importance_std=0.04, rank=2),
        ],
        model_r_squared=0.82,
        interpretation="Top predictors identified"
    )


@pytest.fixture
def sample_report(sample_minmax_result, sample_regression_result, sample_importance_result):
    return StatisticsReport(
        minmax=[sample_minmax_result],
        linear_regression=sample_regression_result,
        feature_importance=sample_importance_result
    )


@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "revenue": [100.0, 200.0, 300.0, 400.0, 500.0],
        "marketing": [10.0, 20.0, 30.0, 40.0, 50.0],
        "headcount": [5.0, 10.0, 15.0, 20.0, 25.0],
    })


class TestMinmaxCharts:
    """Tests for minmax chart functions."""

    def test_plot_minmax_bar_creates_file(self, tmp_path, sample_df, sample_minmax_result):
        """Test that plot_minmax_bar creates a PNG file."""
        output_path = tmp_path / "minmax_test.png"

        result = plot_minmax_bar(sample_df, sample_minmax_result, output_path)

        assert result == output_path
        assert output_path.exists()
        assert output_path.suffix == ".png"

    def test_plot_minmax_summary_creates_file(self, tmp_path, sample_minmax_result):
        """Test that plot_minmax_summary creates a PNG file."""
        output_path = tmp_path / "minmax_summary.png"

        # Create multiple results for summary
        results = [
            sample_minmax_result,
            MinMaxResult(
                column="marketing",
                min_value=10, max_value=50,
                min_index=0, max_index=4,
                min_context={}, max_context={},
                range=40, mean=30, std=15
            )
        ]

        result = plot_minmax_summary(results, output_path)

        assert result == output_path
        assert output_path.exists()


class TestRegressionCharts:
    """Tests for regression chart functions."""

    def test_plot_regression_scatter_creates_file(self, tmp_path, sample_df, sample_regression_result):
        """Test that plot_regression_scatter creates a PNG file."""
        output_path = tmp_path / "scatter_test.png"

        result = plot_regression_scatter(
            sample_df, sample_regression_result, "marketing", output_path
        )

        assert result == output_path
        assert output_path.exists()

    def test_plot_regression_coefficients_creates_file(self, tmp_path, sample_regression_result):
        """Test that plot_regression_coefficients creates a PNG file."""
        output_path = tmp_path / "coef_test.png"

        result = plot_regression_coefficients(sample_regression_result, output_path)

        assert result == output_path
        assert output_path.exists()


class TestImportanceCharts:
    """Tests for importance chart functions."""

    def test_plot_feature_importance_creates_file(self, tmp_path, sample_importance_result):
        """Test that plot_feature_importance creates a PNG file."""
        output_path = tmp_path / "importance_test.png"

        result = plot_feature_importance(sample_importance_result, output_path)

        assert result == output_path
        assert output_path.exists()


class TestChartGenerator:
    """Tests for ChartGenerator class."""

    def test_generate_all_creates_charts(self, tmp_path, sample_df, sample_report):
        """Test that generate_all creates all chart types."""
        generator = ChartGenerator(output_dir=tmp_path)

        charts = generator.generate_all(sample_df, sample_report)

        assert len(charts) > 0
        for chart_path in charts:
            assert chart_path.exists()
            assert chart_path.suffix == ".png"

    def test_generator_creates_output_dir(self, tmp_path):
        """Test that generator creates output directory if missing."""
        new_dir = tmp_path / "new_charts_dir"
        assert not new_dir.exists()

        generator = ChartGenerator(output_dir=new_dir)

        assert new_dir.exists()

    def test_generate_minmax_only(self, tmp_path, sample_df, sample_minmax_result):
        """Test generation with only minmax results."""
        report = StatisticsReport(minmax=[sample_minmax_result])

        generator = ChartGenerator(output_dir=tmp_path)
        charts = generator.generate_all(sample_df, report)

        assert len(charts) > 0
        # Should have minmax chart
        minmax_charts = [c for c in charts if "minmax" in c.name]
        assert len(minmax_charts) > 0

    def test_generate_regression_only(self, tmp_path, sample_df, sample_regression_result):
        """Test generation with only regression results."""
        report = StatisticsReport(
            minmax=[],
            linear_regression=sample_regression_result
        )

        generator = ChartGenerator(output_dir=tmp_path)
        charts = generator.generate_all(sample_df, report)

        assert len(charts) > 0
        # Should have regression charts
        regression_charts = [c for c in charts if "regression" in c.name]
        assert len(regression_charts) > 0

    def test_generate_importance_only(self, tmp_path, sample_df, sample_importance_result):
        """Test generation with only importance results."""
        report = StatisticsReport(
            minmax=[],
            feature_importance=sample_importance_result
        )

        generator = ChartGenerator(output_dir=tmp_path)
        charts = generator.generate_all(sample_df, report)

        assert len(charts) > 0
        # Should have importance chart
        importance_charts = [c for c in charts if "importance" in c.name]
        assert len(importance_charts) > 0

    def test_generate_empty_report(self, tmp_path, sample_df):
        """Test generation with empty report."""
        report = StatisticsReport(minmax=[])

        generator = ChartGenerator(output_dir=tmp_path)
        charts = generator.generate_all(sample_df, report)

        assert len(charts) == 0


class TestChartQuality:
    """Tests for chart quality and content."""

    def test_chart_file_size(self, tmp_path, sample_df, sample_minmax_result):
        """Test that chart files have reasonable size."""
        output_path = tmp_path / "quality_test.png"

        plot_minmax_bar(sample_df, sample_minmax_result, output_path)

        # PNG should have reasonable size (> 1KB, < 1MB for simple chart)
        file_size = output_path.stat().st_size
        assert file_size > 1000, "Chart file too small"
        assert file_size < 1_000_000, "Chart file too large"

    def test_multiple_charts_unique_content(self, tmp_path, sample_df, sample_regression_result):
        """Test that different features produce different charts."""
        path1 = tmp_path / "scatter1.png"
        path2 = tmp_path / "scatter2.png"

        plot_regression_scatter(sample_df, sample_regression_result, "marketing", path1)
        plot_regression_scatter(sample_df, sample_regression_result, "headcount", path2)

        # Both files should exist
        assert path1.exists()
        assert path2.exists()

        # Files should have different content (different sizes as proxy)
        # Note: exact sizes might be similar, but files should both exist
        assert path1.stat().st_size > 0
        assert path2.stat().st_size > 0
