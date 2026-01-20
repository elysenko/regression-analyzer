---
name: 03-stats-engine
status: backlog
created: 2026-01-20T19:38:17Z
progress: 0%
prd: .claude/prds/03-stats-engine.md
github: https://github.com/elysenko/regression-analyzer/issues/9
---

# Epic: 03-stats-engine

## Overview

Implement the statistical analysis engine that provides min/max analysis, linear regression with coefficient significance testing, and Random Forest feature importance using permutation importance. This module converts LLM-identified columns into actionable statistical insights.

## Architecture Decisions

- **Polars for data manipulation**: Consistent with existing codebase, fast operations
- **scikit-learn for ML algorithms**: Industry standard, well-tested implementations
- **Permutation importance over tree importances**: Unbiased, model-agnostic approach (per research findings)
- **scipy for statistical tests**: P-value calculations for coefficient significance
- **Pydantic models for results**: Type-safe, structured output for downstream consumers

## Technical Approach

### Module Structure
```
src/regression_analyzer/stats/
├── __init__.py           # StatsEngine class + exports
├── models.py             # Pydantic result models
├── minmax.py             # Min/max analysis with context
├── regression.py         # Linear regression with coefficient stats
└── feature_importance.py # Random Forest + permutation importance
```

### Core Components
1. **Result Models**: MinMaxResult, LinearRegressionResult, FeatureImportanceResult, StatisticsReport
2. **Min/Max Analysis**: Find extremes with context from other columns
3. **Linear Regression**: Coefficients with p-values and interpretation
4. **Feature Importance**: Permutation-based importance ranking
5. **StatsEngine**: Orchestrator class for full analysis pipeline

### Integration Points
- Receives target/features from LLMAnalyzer (PRD-02)
- Uses DataFrames from DataLoader (PRD-01)
- Outputs structured reports for ChartGenerator (PRD-04)

## Implementation Strategy

The PRD contains complete implementation code. Strategy:
1. Create Pydantic models for all result types
2. Implement statistical functions (minmax, regression, importance)
3. Build StatsEngine orchestrator
4. Add comprehensive tests with known data

## Task Breakdown Preview

- [ ] Task 1: Create stats module with Pydantic models
- [ ] Task 2: Implement min/max analysis and linear regression
- [ ] Task 3: Implement feature importance and StatsEngine orchestrator
- [ ] Task 4: Write comprehensive tests for stats module

## Dependencies

- `polars>=0.20.0` (already in project)
- `scikit-learn>=1.4.0`
- `scipy>=1.11.0`
- `numpy>=1.24.0`
- DataLoader from PRD-01 (completed)
- LLMAnalyzer from PRD-02 (completed)

## Success Criteria (Technical)

- Min/max correctly identifies extremes with row context
- Linear regression R² > 0.95 for perfect linear data
- Coefficient recovery within 10% for known relationships
- P-values correctly identify significant predictors (p < 0.05)
- Feature importance ranks strongest predictor first
- All tests pass with `pytest tests/stats/`

## Estimated Effort

- **Timeline**: 1-2 hours implementation
- **Files to create**: 5 source files + 4 test files
- **Complexity**: Medium (well-specified in PRD, standard sklearn usage)

## Tasks Created
- [ ] #10 - Create stats Pydantic models (parallel: true)
- [ ] #11 - Implement min/max analysis and linear regression (parallel: false)
- [ ] #12 - Implement feature importance and StatsEngine orchestrator (parallel: false)
- [ ] #13 - Write comprehensive tests for stats module (parallel: false)

Total tasks: 4
Parallel tasks: 1
Sequential tasks: 3
Estimated total effort: 2.5 hours
