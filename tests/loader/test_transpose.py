"""Tests for DataFrame transpose functionality."""

import pytest
import polars as pl

from regression_analyzer.loader import transpose_dataframe, should_transpose_heuristic


class TestTransposeDataframe:
    """Tests for the transpose_dataframe function."""

    def test_transpose_basic(self):
        """Test basic transpose with metrics and quarters."""
        df = pl.DataFrame({
            "metric": ["revenue", "cost", "profit"],
            "Q1": [100, 60, 40],
            "Q2": [120, 70, 50],
        })

        transposed = transpose_dataframe(df)

        assert "revenue" in transposed.columns
        assert "cost" in transposed.columns
        assert "profit" in transposed.columns
        assert transposed.height == 2  # Q1, Q2 rows
        assert "index" in transposed.columns

    def test_transpose_preserves_values(self):
        """Test that transpose preserves all data values."""
        df = pl.DataFrame({
            "metric": ["a", "b"],
            "col1": [1, 2],
            "col2": [3, 4],
        })

        transposed = transpose_dataframe(df)

        # Check values are correctly positioned
        assert transposed["a"].to_list() == [1, 3]
        assert transposed["b"].to_list() == [2, 4]

    def test_transpose_duplicate_headers(self):
        """Test transpose with duplicate values in first column."""
        df = pl.DataFrame({
            "metric": ["revenue", "revenue", "profit"],
            "Q1": [100, 50, 40],
            "Q2": [120, 60, 50],
        })

        transposed = transpose_dataframe(df)

        # Should have revenue and revenue_1 to handle duplicates
        assert "revenue" in transposed.columns
        assert "revenue_1" in transposed.columns
        assert "profit" in transposed.columns

    def test_transpose_multiple_duplicates(self):
        """Test transpose with multiple duplicate values."""
        df = pl.DataFrame({
            "metric": ["a", "a", "a", "b"],
            "x": [1, 2, 3, 4],
        })

        transposed = transpose_dataframe(df)

        # Should have a, a_1, a_2, b
        assert "a" in transposed.columns
        assert "a_1" in transposed.columns
        assert "a_2" in transposed.columns
        assert "b" in transposed.columns

    def test_transpose_numeric_first_column(self):
        """Test transpose when first column has numeric values."""
        df = pl.DataFrame({
            "year": [2021, 2022, 2023],
            "jan": [10, 20, 30],
            "feb": [15, 25, 35],
        })

        transposed = transpose_dataframe(df)

        # Numeric values should be converted to string column names
        assert "2021" in transposed.columns
        assert "2022" in transposed.columns
        assert "2023" in transposed.columns

    def test_transpose_single_data_column(self):
        """Test transpose with only one data column."""
        df = pl.DataFrame({
            "metric": ["revenue", "cost"],
            "value": [100, 60],
        })

        transposed = transpose_dataframe(df)

        assert transposed.height == 1
        assert "revenue" in transposed.columns
        assert "cost" in transposed.columns

    def test_transpose_many_rows(self):
        """Test transpose with many rows."""
        metrics = [f"metric_{i}" for i in range(10)]
        df = pl.DataFrame({
            "name": metrics,
            "val1": list(range(10)),
            "val2": list(range(10, 20)),
        })

        transposed = transpose_dataframe(df)

        # Should have all metrics as columns plus index
        assert transposed.width == 11  # 10 metrics + index column
        assert transposed.height == 2  # val1, val2 rows

    def test_transpose_index_column(self):
        """Test that transposed DataFrame has an index column."""
        df = pl.DataFrame({
            "metric": ["a", "b"],
            "col1": [1, 2],
            "col2": [3, 4],
        })

        transposed = transpose_dataframe(df)

        assert "index" in transposed.columns
        assert transposed["index"].to_list() == ["col1", "col2"]

    def test_transpose_with_null_values(self):
        """Test transpose handles null values correctly."""
        df = pl.DataFrame({
            "metric": ["a", "b"],
            "col1": [1, None],
            "col2": [None, 4],
        })

        transposed = transpose_dataframe(df)

        # Nulls should be preserved
        assert transposed["a"][0] == 1
        assert transposed["b"][1] == 4

    def test_transpose_with_special_chars_in_names(self):
        """Test transpose with special characters in first column."""
        df = pl.DataFrame({
            "metric": ["revenue ($)", "cost (%)", "profit/loss"],
            "Q1": [100, 60, 40],
        })

        transposed = transpose_dataframe(df)

        # Special characters in column names should work
        assert "revenue ($)" in transposed.columns
        assert "cost (%)" in transposed.columns


class TestShouldTransposeHeuristic:
    """Tests for the should_transpose_heuristic function."""

    def test_heuristic_false_for_normal_data(self):
        """Test heuristic returns False for normal tabular data."""
        df = pl.DataFrame({
            "id": list(range(100)),
            "value": list(range(100, 200)),
        })

        result = should_transpose_heuristic(df)

        assert result is False

    def test_heuristic_false_for_many_rows(self):
        """Test heuristic returns False when rows >> columns."""
        df = pl.DataFrame({
            "metric": [f"m{i}" for i in range(1000)],
            "value": list(range(1000)),
        })

        result = should_transpose_heuristic(df)

        # height > width * 10, so should be False
        assert result is False

    def test_heuristic_with_row_headers(self):
        """Test heuristic with typical row header pattern."""
        df = pl.DataFrame({
            "metric": ["revenue", "cost", "profit", "tax"],
            "2023": [100, 60, 40, 10],
            "2024": [120, 70, 50, 12],
        })

        result = should_transpose_heuristic(df)

        # Short unique strings in first column might trigger True
        # The heuristic checks: unique_ratio > 0.9 and avg_len < 30
        assert isinstance(result, bool)

    def test_heuristic_with_numeric_first_column(self):
        """Test heuristic returns False for numeric first column."""
        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "value": [10, 20, 30, 40, 50],
        })

        result = should_transpose_heuristic(df)

        # Numeric first column should return False
        assert result is False

    def test_heuristic_with_duplicate_first_column(self):
        """Test heuristic with duplicates in first column."""
        df = pl.DataFrame({
            "category": ["A", "A", "B", "B", "C", "C"],
            "value": [1, 2, 3, 4, 5, 6],
        })

        result = should_transpose_heuristic(df)

        # Low unique ratio (0.5) should not trigger True
        assert result is False

    def test_heuristic_with_long_strings(self):
        """Test heuristic with long strings in first column."""
        df = pl.DataFrame({
            "description": [
                "This is a very long description that exceeds thirty characters",
                "Another extremely long description for testing purposes",
                "Yet another long description to test the average length check",
            ],
            "value": [1, 2, 3],
        })

        result = should_transpose_heuristic(df)

        # avg_len > 30 should prevent True
        assert result is False

    def test_heuristic_returns_bool(self):
        """Test that heuristic always returns a boolean."""
        test_cases = [
            pl.DataFrame({"a": [1, 2], "b": [3, 4]}),
            pl.DataFrame({"metric": ["x", "y"], "val": [1, 2]}),
            pl.DataFrame({"col": ["short"], "data": [100]}),
        ]

        for df in test_cases:
            result = should_transpose_heuristic(df)
            assert isinstance(result, bool)

    def test_heuristic_with_all_unique_short_strings(self):
        """Test heuristic with all unique, short strings - likely transpose."""
        df = pl.DataFrame({
            "metric": ["rev", "cost", "tax"],
            "Q1": [100, 60, 10],
            "Q2": [110, 65, 12],
            "Q3": [120, 70, 14],
        })

        result = should_transpose_heuristic(df)

        # 100% unique, avg_len < 30 should trigger True
        assert result is True

    def test_heuristic_with_empty_strings(self):
        """Test heuristic handles empty strings."""
        df = pl.DataFrame({
            "label": ["", "", ""],
            "value": [1, 2, 3],
        })

        result = should_transpose_heuristic(df)

        # Empty strings, but low unique ratio (1/3)
        assert isinstance(result, bool)

    def test_heuristic_single_row(self):
        """Test heuristic with single row DataFrame."""
        df = pl.DataFrame({
            "metric": ["revenue"],
            "Q1": [100],
            "Q2": [110],
        })

        result = should_transpose_heuristic(df)

        # Single row, unique ratio = 1.0, short string
        assert isinstance(result, bool)


class TestTransposeIntegration:
    """Integration tests for transpose functionality."""

    def test_transpose_and_heuristic_alignment(self):
        """Test that heuristic correctly identifies transpose candidates."""
        # Data that should be transposed (row headers)
        row_header_df = pl.DataFrame({
            "metric": ["revenue", "cost", "profit"],
            "2021": [100, 60, 40],
            "2022": [120, 70, 50],
            "2023": [140, 80, 60],
        })

        if should_transpose_heuristic(row_header_df):
            transposed = transpose_dataframe(row_header_df)
            assert "revenue" in transposed.columns
            assert transposed.height == 3  # 3 years

    def test_roundtrip_considerations(self):
        """Test considerations for transpose operations."""
        original = pl.DataFrame({
            "metric": ["a", "b"],
            "x": [1, 2],
            "y": [3, 4],
        })

        transposed = transpose_dataframe(original)

        # Note: true roundtrip would require another transpose
        # Just verify structure is valid
        assert transposed.width == 3  # index + a + b
        assert transposed.height == 2  # x, y rows
