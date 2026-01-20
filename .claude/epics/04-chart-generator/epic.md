---
name: 04-chart-generator
status: backlog
created: 2026-01-20T19:46:40Z
progress: 0%
prd: .claude/prds/04-chart-generator.md
github: https://github.com/elysenko/regression-analyzer/issues/14
---

# Epic: 04-chart-generator

## Overview

Implement PNG chart generation for min/max visualization, regression plots, and feature importance. Uses matplotlib and seaborn for professional-quality static charts.

## Architecture Decisions

- **Seaborn for styling**: Best static chart quality with minimal code
- **matplotlib backend**: Reliable PNG export
- **Consistent color palette**: Professional appearance
- **Agg backend**: Headless server compatibility

## Technical Approach

### Module Structure
```
src/regression_analyzer/charts/
├── __init__.py           # ChartGenerator class + exports
├── generator.py          # Orchestrator class
├── styles.py             # Color palette and figure sizes
├── minmax_charts.py      # Min/max bar charts
├── regression_charts.py  # Scatter plots with fit lines
└── importance_charts.py  # Feature importance bars
```

### Core Components
1. **Styles**: Color palette, figure sizes, seaborn/matplotlib config
2. **Min/Max Charts**: Bar charts with min/max annotations and mean lines
3. **Regression Charts**: Scatter plots with fit lines, coefficient bars
4. **Importance Charts**: Horizontal bars with error bars
5. **ChartGenerator**: Orchestrator for full report visualization

### Integration Points
- Receives StatisticsReport from StatsEngine (PRD-03)
- Uses DataFrame from DataLoader (PRD-01)
- Outputs PNG files for CLI display (PRD-05)

## Implementation Strategy

The PRD contains complete implementation code. Strategy:
1. Create styles module with color palette and configuration
2. Implement all chart types (minmax, regression, importance)
3. Build ChartGenerator orchestrator
4. Add comprehensive tests

## Task Breakdown Preview

- [ ] Task 1: Create styles and minmax charts
- [ ] Task 2: Implement regression and importance charts
- [ ] Task 3: Implement ChartGenerator orchestrator and tests

## Dependencies

- `matplotlib>=3.8.0` (already in project)
- `seaborn>=0.13.0` (already in project)
- `numpy>=1.24.0` (already in project)
- StatsEngine from PRD-03 (completed)

## Success Criteria (Technical)

- Min/max charts display annotations correctly
- Regression scatter plots show fit lines and R²
- Feature importance shows ranked bars with error bars
- All charts export as PNG at 150 DPI
- Charts are readable and professional

## Estimated Effort

- **Timeline**: 1 hour implementation
- **Files to create**: 6 source files + 1 test file
- **Complexity**: Medium (well-specified in PRD)
