"""Tests for analyzer prompts module."""

import pytest
import polars as pl

from regression_analyzer.analyzer.prompts import (
    format_table_as_markdown,
    CONTEXT_IDENTIFICATION_PROMPT,
    COLUMN_RELEVANCE_PROMPT,
    HEADER_ANALYSIS_PROMPT,
)


class TestFormatTableAsMarkdown:
    """Tests for format_table_as_markdown function."""

    def test_format_table_as_markdown_basic(self):
        """Test basic markdown table formatting."""
        df = pl.DataFrame({
            "name": ["Alice", "Bob"],
            "age": [30, 25]
        })
        result = format_table_as_markdown(df)

        assert "| name | age |" in result
        assert "| --- | --- |" in result
        assert "Alice" in result
        assert "Bob" in result

    def test_format_table_as_markdown_max_rows(self):
        """Test that max_rows limits output."""
        df = pl.DataFrame({"x": list(range(100))})
        result = format_table_as_markdown(df, max_rows=5)

        lines = result.strip().split('\n')
        # header + separator + 5 data rows = 7 lines
        assert len(lines) == 7

    def test_format_table_as_markdown_empty(self):
        """Test formatting empty DataFrame (no rows)."""
        df = pl.DataFrame({"a": []}).cast({"a": pl.Int64})
        result = format_table_as_markdown(df)

        assert "| a |" in result
        assert "| --- |" in result

    def test_format_table_as_markdown_single_row(self):
        """Test formatting DataFrame with single row."""
        df = pl.DataFrame({
            "col1": ["value1"],
            "col2": [123]
        })
        result = format_table_as_markdown(df)

        assert "| col1 | col2 |" in result
        assert "value1" in result
        assert "123" in result

    def test_format_table_as_markdown_multiple_columns(self):
        """Test formatting DataFrame with many columns."""
        df = pl.DataFrame({
            "a": [1],
            "b": [2],
            "c": [3],
            "d": [4],
            "e": [5]
        })
        result = format_table_as_markdown(df)

        assert "| a | b | c | d | e |" in result
        # Should have 5 separators
        assert result.count("---") == 5

    def test_format_table_as_markdown_with_nulls(self):
        """Test formatting DataFrame containing null values."""
        df = pl.DataFrame({
            "name": ["Alice", None, "Charlie"],
            "score": [100, 85, None]
        })
        result = format_table_as_markdown(df)

        # Nulls are represented as "null" in polars string conversion
        assert "Alice" in result
        assert "null" in result or "None" in result

    def test_format_table_as_markdown_with_special_characters(self):
        """Test formatting with special characters in data."""
        df = pl.DataFrame({
            "text": ["hello|world", "pipe|in|text"],
            "num": [1, 2]
        })
        result = format_table_as_markdown(df)

        # Should contain the data even with pipes
        assert "hello|world" in result

    def test_format_table_as_markdown_default_max_rows(self):
        """Test that default max_rows is 10."""
        df = pl.DataFrame({"x": list(range(20))})
        result = format_table_as_markdown(df)  # no max_rows specified

        lines = result.strip().split('\n')
        # header + separator + 10 data rows = 12 lines
        assert len(lines) == 12

    def test_format_table_as_markdown_max_rows_larger_than_data(self):
        """Test max_rows larger than actual data."""
        df = pl.DataFrame({"x": [1, 2, 3]})
        result = format_table_as_markdown(df, max_rows=100)

        lines = result.strip().split('\n')
        # header + separator + 3 data rows = 5 lines
        assert len(lines) == 5

    def test_format_table_as_markdown_numeric_types(self):
        """Test formatting various numeric types."""
        df = pl.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
        })
        result = format_table_as_markdown(df)

        assert "1.5" in result
        assert "2.5" in result

    def test_format_table_as_markdown_date_types(self):
        """Test formatting date/datetime columns."""
        from datetime import date
        df = pl.DataFrame({
            "date_col": [date(2024, 1, 1), date(2024, 2, 1)],
        })
        result = format_table_as_markdown(df)

        assert "2024-01-01" in result
        assert "2024-02-01" in result


class TestPromptTemplates:
    """Tests for prompt template strings."""

    def test_context_identification_prompt_has_placeholders(self):
        """Test that CONTEXT_IDENTIFICATION_PROMPT has required placeholders."""
        assert "{table_markdown}" in CONTEXT_IDENTIFICATION_PROMPT
        assert "{column_info}" in CONTEXT_IDENTIFICATION_PROMPT

    def test_column_relevance_prompt_has_placeholders(self):
        """Test that COLUMN_RELEVANCE_PROMPT has required placeholders."""
        assert "{context}" in COLUMN_RELEVANCE_PROMPT
        assert "{table_markdown}" in COLUMN_RELEVANCE_PROMPT
        assert "{column_list}" in COLUMN_RELEVANCE_PROMPT

    def test_header_analysis_prompt_has_placeholders(self):
        """Test that HEADER_ANALYSIS_PROMPT has required placeholders."""
        assert "{table_head_markdown}" in HEADER_ANALYSIS_PROMPT
        assert "{first_column_values}" in HEADER_ANALYSIS_PROMPT
        assert "{first_row_values}" in HEADER_ANALYSIS_PROMPT

    def test_context_identification_prompt_mentions_json(self):
        """Test that CONTEXT_IDENTIFICATION_PROMPT requests JSON output."""
        assert "JSON" in CONTEXT_IDENTIFICATION_PROMPT or "json" in CONTEXT_IDENTIFICATION_PROMPT.lower()

    def test_column_relevance_prompt_mentions_json(self):
        """Test that COLUMN_RELEVANCE_PROMPT requests JSON output."""
        assert "JSON" in COLUMN_RELEVANCE_PROMPT or "json" in COLUMN_RELEVANCE_PROMPT.lower()

    def test_header_analysis_prompt_mentions_json(self):
        """Test that HEADER_ANALYSIS_PROMPT requests JSON output."""
        assert "JSON" in HEADER_ANALYSIS_PROMPT or "json" in HEADER_ANALYSIS_PROMPT.lower()

    def test_context_identification_prompt_can_be_formatted(self):
        """Test that CONTEXT_IDENTIFICATION_PROMPT can be formatted without error."""
        formatted = CONTEXT_IDENTIFICATION_PROMPT.format(
            table_markdown="| col1 | col2 |\n| --- | --- |\n| val1 | val2 |",
            column_info="- col1: String\n- col2: Int64"
        )
        assert "col1" in formatted
        assert "col2" in formatted

    def test_column_relevance_prompt_can_be_formatted(self):
        """Test that COLUMN_RELEVANCE_PROMPT can be formatted without error."""
        formatted = COLUMN_RELEVANCE_PROMPT.format(
            context="Company type: SaaS\nIndustry: tech",
            table_markdown="| col1 |\n| --- |\n| val1 |",
            column_list="- col1 (String)"
        )
        assert "SaaS" in formatted
        assert "col1" in formatted

    def test_header_analysis_prompt_can_be_formatted(self):
        """Test that HEADER_ANALYSIS_PROMPT can be formatted without error."""
        formatted = HEADER_ANALYSIS_PROMPT.format(
            table_head_markdown="| a | b |\n| --- | --- |",
            first_column_values="['x', 'y']",
            first_row_values="['a', 'b']"
        )
        assert "['x', 'y']" in formatted
