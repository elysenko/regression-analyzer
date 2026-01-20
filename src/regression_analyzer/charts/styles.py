# src/regression_analyzer/charts/styles.py

import matplotlib.pyplot as plt
import seaborn as sns

# Color palette
COLORS = {
    "primary": "#2563eb",    # Blue
    "secondary": "#7c3aed",  # Purple
    "success": "#059669",    # Green
    "warning": "#d97706",    # Orange
    "danger": "#dc2626",     # Red
    "neutral": "#6b7280",    # Gray
    "min": "#dc2626",        # Red for minimums
    "max": "#059669",        # Green for maximums
}

# Figure sizes
FIGURE_SIZES = {
    "small": (8, 6),
    "medium": (10, 8),
    "large": (12, 9),
    "wide": (14, 6),
}


def setup_style():
    """Configure global matplotlib/seaborn style."""
    # Use Agg backend for headless servers
    plt.switch_backend('Agg')
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def get_color(name: str) -> str:
    """Get color by name."""
    return COLORS.get(name, COLORS["neutral"])
