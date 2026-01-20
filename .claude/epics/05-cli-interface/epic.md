---
name: 05-cli-interface
status: backlog
created: 2026-01-20T19:58:14Z
progress: 0%
prd: .claude/prds/05-cli-interface.md
github: https://github.com/elysenko/regression-analyzer/issues/18
---

# Epic: 05-cli-interface

## Overview

Implement the command-line interface for running the complete analysis pipeline. Uses Click for CLI framework and Rich for terminal formatting.

## Architecture Decisions

- **Click for CLI**: Simple, Pythonic CLI framework
- **Rich for output**: Beautiful terminal output with progress indicators
- **Async pipeline**: Support for async LLM calls
- **JSON output option**: Programmatic usage support

## Technical Approach

### Module Structure
```
src/regression_analyzer/
├── cli.py       # Click command definitions
└── pipeline.py  # Analysis orchestration
```

### Core Components
1. **CLI Commands**: analyze (main), providers
2. **Rich Output**: Progress spinners, tables, panels
3. **AnalysisPipeline**: Orchestrate all modules
4. **AnalysisResult**: Aggregate results for output

### Integration Points
- Uses DataLoader (PRD-01)
- Uses LLMAnalyzer (PRD-02)
- Uses StatsEngine (PRD-03)
- Uses ChartGenerator (PRD-04)

## Implementation Strategy

The PRD contains complete implementation code. Strategy:
1. Create CLI with Click commands
2. Implement AnalysisPipeline orchestrator
3. Add Rich output formatting
4. Write comprehensive tests

## Task Breakdown Preview

- [ ] Task 1: Create CLI and pipeline
- [ ] Task 2: Add tests

## Dependencies

- `click>=8.0.0` (already in project)
- `rich>=13.0.0` (already in project)
- All core modules (completed)

## Success Criteria (Technical)

- `regression-analyzer analyze file.csv` works
- `--skip-llm` mode runs statistics only
- `--json-output` produces valid JSON
- Progress indicators shown during analysis
- Results displayed in formatted tables

## Estimated Effort

- **Timeline**: 1 hour implementation
- **Files to create**: 2 source files + 2 test files
- **Complexity**: Medium
