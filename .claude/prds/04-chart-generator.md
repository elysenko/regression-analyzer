---
name: chart-generator
id: PRD-04
description: PNG chart generation for min/max visualization and regression plots
status: backlog
phase: mvp
priority: P1
complexity: medium
wave: 4

depends_on:
  - PRD-03  # Needs stats results to visualize

creates:
  - src/regression_analyzer/charts/__init__.py
  - src/regression_analyzer/charts/generator.py
  - src/regression_analyzer/charts/minmax_charts.py
  - src/regression_analyzer/charts/regression_charts.py
  - src/regression_analyzer/charts/importance_charts.py
  - src/regression_analyzer/charts/styles.py
  - tests/charts/test_generator.py

modifies: []

database:
  creates: []
  modifies: []

test_command: pytest tests/charts/

blocks: [PRD-05]

references: [PRD-03]

created: 2026-01-20T19:06:34Z
updated: 2026-01-20T19:06:34Z
---

# PRD-04: Chart Generator

## Overview

**Feature:** PNG chart generation for min/max visualization and regression plots
**Priority:** P1 (Visualization layer)
**Complexity:** Medium
**Dependencies:** PRD-03 (Stats Engine)

---

## Problem Statement

Statistical results need visual representation to communicate insights effectively:
1. Min/max values need context visualization
2. Regression results need scatter plots with fit lines
3. Feature importance needs bar charts for comparison
4. All charts must export as PNG for reports and CLI display

Research findings:
- Seaborn provides best static chart quality with minimal code
- matplotlib backend for PNG export
- Consistent styling improves professionalism

---

## Goals

1. Generate min/max highlight charts (bar or line with annotations)
2. Generate scatter plots with regression lines
3. Generate feature importance bar charts
4. Export all charts as PNG files
5. Consistent, professional styling
6. Support both individual charts and combined dashboards

---

## Non-Goals

- Interactive charts (Plotly for MVP is optional)
- Real-time updating charts
- Web embedding
- Custom themes beyond defaults

---

## Technical Design

### Architecture

```
charts/
├── __init__.py           # Public API exports
├── generator.py          # Main ChartGenerator class
├── minmax_charts.py      # Min/max visualizations
├── regression_charts.py  # Regression plots
├── importance_charts.py  # Feature importance charts
└── styles.py            # Shared styling configuration
```

### Shared Styles

```python
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
```

### Min/Max Charts

```python
# src/regression_analyzer/charts/minmax_charts.py

from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
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
```

### Regression Charts

```python
# src/regression_analyzer/charts/regression_charts.py

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
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
    from matplotlib.patches import Patch
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
```

### Feature Importance Charts

```python
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

    ax.bar(x, importances, yerr=stds, color=get_color("primary"),
           alpha=0.8, capsize=3, ecolor='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Importance')
    ax.set_title(title)

    # Highlight top 3
    for i in range(min(3, len(features))):
        ax.get_children()[i].set_color(get_color("success"))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    return output_path
```

### Chart Generator Orchestrator

```python
# src/regression_analyzer/charts/generator.py

from pathlib import Path
from typing import List, Optional
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
```

---

## Implementation Tasks

### Task 1: Styles Module
- [ ] Create `styles.py` with color palette
- [ ] Configure matplotlib/seaborn defaults
- [ ] Define figure sizes

### Task 2: Min/Max Charts
- [ ] Implement `plot_minmax_bar()` with annotations
- [ ] Implement `plot_minmax_summary()` for overview
- [ ] Test with sample data

### Task 3: Regression Charts
- [ ] Implement `plot_regression_scatter()` with fit line
- [ ] Implement `plot_regression_coefficients()` bar chart
- [ ] Add significance coloring

### Task 4: Feature Importance Charts
- [ ] Implement `plot_feature_importance()` horizontal bars
- [ ] Add error bars from permutation std
- [ ] Gradient coloring by importance

### Task 5: Chart Generator
- [ ] Create `ChartGenerator` orchestrator
- [ ] Auto-generate all charts from report
- [ ] Handle output directory creation

### Task 6: Testing
- [ ] Test chart generation with mock data
- [ ] Verify PNG files are created
- [ ] Visual inspection of output quality

---

## Testing Strategy

### Unit Tests
```python
# tests/charts/test_generator.py

import pytest
from pathlib import Path
import polars as pl

from regression_analyzer.charts import ChartGenerator
from regression_analyzer.stats.models import (
    StatisticsReport, MinMaxResult, LinearRegressionResult,
    RegressionCoefficient, FeatureImportanceResult, FeatureImportance
)

@pytest.fixture
def sample_report():
    return StatisticsReport(
        minmax=[
            MinMaxResult(
                column="revenue",
                min_value=100, max_value=500,
                min_index=0, max_index=9,
                min_context={}, max_context={},
                range=400, mean=300, std=100
            )
        ],
        linear_regression=LinearRegressionResult(
            target="revenue",
            features=["marketing", "headcount"],
            coefficients=[
                RegressionCoefficient(
                    feature="marketing", coefficient=2.5,
                    p_value=0.01, is_significant=True
                ),
                RegressionCoefficient(
                    feature="headcount", coefficient=10.0,
                    p_value=0.001, is_significant=True
                ),
            ],
            intercept=50.0, r_squared=0.85,
            adjusted_r_squared=0.83, rmse=20.0,
            n_samples=100, interpretation="Strong fit"
        ),
        feature_importance=FeatureImportanceResult(
            target="revenue",
            features=[
                FeatureImportance(feature="headcount", importance=0.4, importance_std=0.05, rank=1),
                FeatureImportance(feature="marketing", importance=0.3, importance_std=0.04, rank=2),
            ],
            model_r_squared=0.82, interpretation="Top predictors identified"
        )
    )

def test_generate_all_charts(tmp_path, sample_report):
    """Test that all chart types are generated."""
    df = pl.DataFrame({
        "revenue": [100, 200, 300, 400, 500],
        "marketing": [10, 20, 30, 40, 50],
        "headcount": [5, 10, 15, 20, 25],
    })

    generator = ChartGenerator(output_dir=tmp_path)
    charts = generator.generate_all(df, sample_report)

    assert len(charts) > 0
    for chart_path in charts:
        assert chart_path.exists()
        assert chart_path.suffix == ".png"
```

---

## Acceptance Criteria

1. **Min/Max Charts**: Bar charts with min/max annotations
2. **Regression Charts**: Scatter plots with fit lines, R² displayed
3. **Coefficient Charts**: Horizontal bars with significance colors
4. **Importance Charts**: Ranked bars with error bars
5. **PNG Export**: All charts save as PNG at 150 DPI
6. **Professional Style**: Consistent colors, fonts, layouts

---

## Dependencies

- `matplotlib>=3.8.0` - Core plotting
- `seaborn>=0.13.0` - Statistical visualizations
- `numpy>=1.24.0` - Array operations

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Display issues on headless servers | Use Agg backend, no plt.show() |
| Large datasets slow rendering | Sample data for scatter plots |
| Color accessibility | Use colorblind-friendly palette |
