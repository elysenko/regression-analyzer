# src/regression_analyzer/charts/regression_charts.py

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import polars as pl

from ..stats.models import LinearRegressionResult
from .styles import setup_style, get_color, FIGURE_SIZES


def plot_regression_scatter(
    df: pl.DataFrame,
    result: LinearRegressionResult,
    feature: str,
    output_path: Path,
    title: Optional[str] = None
) -> Path:
    """Create scatter plot with regression line for single feature.

    Args:
        df: Source DataFrame
        result: Regression result
        feature: Feature column to plot
        output_path: Path to save PNG
        title: Optional chart title

    Returns:
        Path to saved chart
    """
    setup_style()

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["medium"])

    x = df[feature].to_numpy()
    y = df[result.target].to_numpy()

    # Scatter plot
    ax.scatter(x, y, alpha=0.6, color=get_color("primary"), label='Data')

    # Regression line
    coef = next((c for c in result.coefficients if c.feature == feature), None)
    if coef:
        x_line = np.linspace(x.min(), x.max(), 100)
        # Simple univariate approximation for visualization
        y_line = result.intercept + coef.coefficient * x_line
        ax.plot(x_line, y_line, color=get_color("danger"), linewidth=2,
                label=f'Fit (coef={coef.coefficient:.3f})')

    ax.set_xlabel(feature)
    ax.set_ylabel(result.target)
    ax.set_title(title or f'{result.target} vs {feature}')
    ax.legend()

    # Add R² annotation
    ax.annotate(
        f'R² = {result.r_squared:.3f}',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_regression_coefficients(
    result: LinearRegressionResult,
    output_path: Path,
    title: str = "Regression Coefficients"
) -> Path:
    """Create bar chart of regression coefficients.

    Args:
        result: Regression result
        output_path: Path to save PNG
        title: Chart title

    Returns:
        Path to saved chart
    """
    setup_style()

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["medium"])

    features = [c.feature for c in result.coefficients]
    coefs = [c.coefficient for c in result.coefficients]
    significant = [c.is_significant for c in result.coefficients]

    # Color by significance
    colors = [
        get_color("success") if sig else get_color("neutral")
        for sig in significant
    ]

    y_pos = range(len(features))
    bars = ax.barh(y_pos, coefs, color=colors, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Coefficient')
    ax.set_title(title)
    ax.axvline(x=0, color='black', linewidth=0.5)

    # Add significance legend
    legend_elements = [
        Patch(facecolor=get_color("success"), label='Significant (p<0.05)'),
        Patch(facecolor=get_color("neutral"), label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_actual_vs_predicted(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    output_path: Path,
    title: str = "Actual vs Predicted"
) -> Path:
    """Create actual vs predicted scatter plot.

    Args:
        y_actual: Actual values
        y_predicted: Predicted values
        output_path: Path to save PNG
        title: Chart title

    Returns:
        Path to saved chart
    """
    setup_style()

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["small"])

    ax.scatter(y_actual, y_predicted, alpha=0.6, color=get_color("primary"))

    # Perfect prediction line
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect fit')

    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    return output_path
