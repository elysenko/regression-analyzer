# src/regression_analyzer/charts/generator.py

from pathlib import Path
from typing import List
import polars as pl

from ..stats.models import StatisticsReport
from .minmax_charts import plot_minmax_bar, plot_minmax_summary
from .regression_charts import (
    plot_regression_scatter,
    plot_regression_coefficients,
)
from .importance_charts import plot_feature_importance


class ChartGenerator:
    """Generate all charts from statistics report."""

    def __init__(self, output_dir: str | Path = "./charts"):
        """Initialize chart generator.

        Args:
            output_dir: Directory to save charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(
        self,
        df: pl.DataFrame,
        report: StatisticsReport
    ) -> List[Path]:
        """Generate all charts from statistics report.

        Args:
            df: Source DataFrame
            report: Statistics report

        Returns:
            List of paths to generated charts
        """
        charts = []

        # Min/max charts
        if report.minmax:
            charts.extend(self._generate_minmax_charts(df, report.minmax))

        # Regression charts
        if report.linear_regression:
            charts.extend(self._generate_regression_charts(df, report.linear_regression))

        # Feature importance charts
        if report.feature_importance:
            charts.extend(self._generate_importance_charts(report.feature_importance))

        return charts

    def _generate_minmax_charts(
        self,
        df: pl.DataFrame,
        minmax_results: List
    ) -> List[Path]:
        """Generate min/max visualizations."""
        charts = []

        # Summary chart
        if len(minmax_results) > 1:
            path = self.output_dir / "minmax_summary.png"
            plot_minmax_summary(minmax_results, path)
            charts.append(path)

        # Individual column charts (top 3 by range)
        sorted_results = sorted(minmax_results, key=lambda x: x.range, reverse=True)
        for result in sorted_results[:3]:
            path = self.output_dir / f"minmax_{result.column}.png"
            plot_minmax_bar(df, result, path)
            charts.append(path)

        return charts

    def _generate_regression_charts(
        self,
        df: pl.DataFrame,
        result
    ) -> List[Path]:
        """Generate regression visualizations."""
        charts = []

        # Coefficient chart
        path = self.output_dir / "regression_coefficients.png"
        plot_regression_coefficients(result, path)
        charts.append(path)

        # Scatter plots for top features
        significant = [c for c in result.coefficients if c.is_significant]
        for coef in significant[:3]:
            path = self.output_dir / f"regression_scatter_{coef.feature}.png"
            plot_regression_scatter(df, result, coef.feature, path)
            charts.append(path)

        return charts

    def _generate_importance_charts(
        self,
        result
    ) -> List[Path]:
        """Generate feature importance visualizations."""
        charts = []

        path = self.output_dir / "feature_importance.png"
        plot_feature_importance(result, path)
        charts.append(path)

        return charts
