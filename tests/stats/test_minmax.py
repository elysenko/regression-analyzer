# tests/stats/test_minmax.py

import pytest
import polars as pl
import numpy as np

from regression_analyzer.stats import analyze_minmax
from regression_analyzer.stats.models import MinMaxResult


class TestAnalyzeMinmax:
    """Tests for min/max analysis."""

    def test_minmax_simple(self):
        """Test basic min/max identification."""
        df = pl.DataFrame({
            "values": [10.0, 5.0, 20.0, 15.0],
            "labels": ["a", "b", "c", "d"]
        })

        results = analyze_minmax(df, ["values"])

        assert len(results) == 1
        result = results[0]
        assert result.column == "values"
        assert result.min_value == 5.0
        assert result.max_value == 20.0
        assert result.min_index == 1
        assert result.max_index == 2

    def test_minmax_with_context(self):
        """Test that context from other columns is captured."""
        df = pl.DataFrame({
            "sales": [100.0, 50.0, 200.0],
            "region": ["North", "South", "East"],
            "quarter": ["Q1", "Q2", "Q3"]
        })

        results = analyze_minmax(df, ["sales"])
        result = results[0]

        # Min at index 1 (South, Q2)
        assert result.min_context["region"] == "South"
        assert result.min_context["quarter"] == "Q2"

        # Max at index 2 (East, Q3)
        assert result.max_context["region"] == "East"
        assert result.max_context["quarter"] == "Q3"

    def test_minmax_statistics(self):
        """Test calculated statistics (mean, std, range)."""
        df = pl.DataFrame({
            "values": [10.0, 20.0, 30.0, 40.0]
        })

        results = analyze_minmax(df)
        result = results[0]

        assert result.range == 30.0
        assert result.mean == 25.0
        # std should be close to 12.91 for this data
        assert abs(result.std - 12.91) < 0.1

    def test_minmax_auto_detect_numeric(self):
        """Test that numeric columns are auto-detected."""
        df = pl.DataFrame({
            "num1": [1.0, 2.0, 3.0],
            "num2": [10, 20, 30],
            "text": ["a", "b", "c"]
        })

        results = analyze_minmax(df)  # No columns specified

        # Should find 2 numeric columns
        assert len(results) == 2
        columns = {r.column for r in results}
        assert "num1" in columns
        assert "num2" in columns
        assert "text" not in columns

    def test_minmax_skip_all_null(self):
        """Test that all-null columns are skipped."""
        df = pl.DataFrame({
            "valid": [1.0, 2.0, 3.0],
            "all_null": [None, None, None]
        }).cast({"all_null": pl.Float64})

        results = analyze_minmax(df)

        assert len(results) == 1
        assert results[0].column == "valid"

    def test_minmax_multiple_columns(self):
        """Test analysis of multiple columns."""
        df = pl.DataFrame({
            "sales": [100.0, 200.0, 150.0],
            "profit": [10.0, 30.0, 20.0],
            "units": [5, 10, 7]
        })

        results = analyze_minmax(df)

        assert len(results) == 3

        sales_result = next(r for r in results if r.column == "sales")
        assert sales_result.min_value == 100.0
        assert sales_result.max_value == 200.0
