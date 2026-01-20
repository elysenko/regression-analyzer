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
