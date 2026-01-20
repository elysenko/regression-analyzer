# src/regression_analyzer/charts/importance_charts.py

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

from ..stats.models import FeatureImportanceResult
from .styles import setup_style, get_color, FIGURE_SIZES


def plot_feature_importance(
    result: FeatureImportanceResult,
    output_path: Path,
    top_n: int = 10,
    title: Optional[str] = None
) -> Path:
    """Create horizontal bar chart of feature importance.

    Args:
        result: Feature importance result
        output_path: Path to save PNG
        top_n: Number of top features to show
        title: Optional chart title

    Returns:
        Path to saved chart
    """
    setup_style()

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["medium"])

    # Get top N features
    features = result.features[:top_n]
    names = [f.feature for f in features]
    importances = [f.importance for f in features]
    stds = [f.importance_std for f in features]

    # Reverse for horizontal bar (top at top)
    names = names[::-1]
    importances = importances[::-1]
    stds = stds[::-1]

    y_pos = range(len(names))

    # Create gradient colors based on importance
    max_imp = max(importances) if importances else 1
    colors = [
        plt.cm.Blues(0.3 + 0.7 * (imp / max_imp))
        for imp in importances
    ]

    bars = ax.barh(y_pos, importances, xerr=stds, color=colors, alpha=0.8,
                   capsize=3, ecolor='gray')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Importance (permutation)')
    ax.set_title(title or f'Feature Importance for {result.target}')

    # Add value labels
    for i, (imp, std) in enumerate(zip(importances, stds)):
        ax.annotate(
            f'{imp:.3f}',
            xy=(imp + std + 0.002, i),
            va='center',
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_importance_comparison(
    result: FeatureImportanceResult,
    output_path: Path,
    title: str = "Feature Importance Comparison"
) -> Path:
    """Create comparison chart with importance and error bars.

    Args:
        result: Feature importance result
        output_path: Path to save PNG
        title: Chart title

    Returns:
        Path to saved chart
    """
    setup_style()

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["wide"])

    features = result.features
    names = [f.feature for f in features]
    importances = [f.importance for f in features]
    stds = [f.importance_std for f in features]

    x = range(len(names))

    bars = ax.bar(x, importances, yerr=stds, color=get_color("primary"),
                  alpha=0.8, capsize=3, ecolor='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Importance')
    ax.set_title(title)

    # Highlight top 3
    for i in range(min(3, len(features))):
        bars[i].set_color(get_color("success"))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    return output_path
