# src/regression_analyzer/charts/minmax_charts.py

from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import polars as pl

from ..stats.models import MinMaxResult
from .styles import setup_style, get_color, FIGURE_SIZES


def plot_minmax_bar(
    df: pl.DataFrame,
    minmax_result: MinMaxResult,
    output_path: Path,
    title: Optional[str] = None
) -> Path:
    """Create bar chart highlighting min/max values.

    Args:
        df: Source DataFrame
        minmax_result: MinMax analysis result
        output_path: Path to save PNG
        title: Optional chart title

    Returns:
        Path to saved chart
    """
    setup_style()

    col = minmax_result.column
    values = df[col].to_list()

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["wide"])

    # Create bar colors
    colors = []
    for i, val in enumerate(values):
        if i == minmax_result.min_index:
            colors.append(get_color("min"))
        elif i == minmax_result.max_index:
            colors.append(get_color("max"))
        else:
            colors.append(get_color("neutral"))

    # Plot bars
    x = range(len(values))
    bars = ax.bar(x, values, color=colors, alpha=0.8)

    # Annotate min/max
    ax.annotate(
        f'MIN: {minmax_result.min_value:.2f}',
        xy=(minmax_result.min_index, minmax_result.min_value),
        xytext=(0, -20),
        textcoords='offset points',
        ha='center',
        fontsize=9,
        color=get_color("min"),
        fontweight='bold'
    )

    ax.annotate(
        f'MAX: {minmax_result.max_value:.2f}',
        xy=(minmax_result.max_index, minmax_result.max_value),
        xytext=(0, 10),
        textcoords='offset points',
        ha='center',
        fontsize=9,
        color=get_color("max"),
        fontweight='bold'
    )

    # Add mean line
    ax.axhline(
        y=minmax_result.mean,
        color=get_color("primary"),
        linestyle='--',
        label=f'Mean: {minmax_result.mean:.2f}'
    )

    ax.set_xlabel('Index')
    ax.set_ylabel(col)
    ax.set_title(title or f'{col}: Min/Max Analysis')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_minmax_summary(
    minmax_results: List[MinMaxResult],
    output_path: Path,
    title: str = "Min/Max Summary"
) -> Path:
    """Create summary chart showing all columns' ranges.

    Args:
        minmax_results: List of MinMax results
        output_path: Path to save PNG
        title: Chart title

    Returns:
        Path to saved chart
    """
    setup_style()

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["medium"])

    columns = [r.column for r in minmax_results]
    mins = [r.min_value for r in minmax_results]
    maxs = [r.max_value for r in minmax_results]
    means = [r.mean for r in minmax_results]

    x = range(len(columns))
    width = 0.25

    ax.bar([i - width for i in x], mins, width, label='Min', color=get_color("min"), alpha=0.8)
    ax.bar(x, means, width, label='Mean', color=get_color("primary"), alpha=0.8)
    ax.bar([i + width for i in x], maxs, width, label='Max', color=get_color("max"), alpha=0.8)

    ax.set_xlabel('Columns')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(columns, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    return output_path
