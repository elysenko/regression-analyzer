---
name: 02-llm-analyzer
status: backlog
created: 2026-01-20T19:25:35Z
progress: 0%
prd: .claude/prds/02-llm-analyzer.md
github: https://github.com/elysenko/regression-analyzer/issues/5
---

# Epic: 02-llm-analyzer

## Overview

Implement the LLM-powered analysis module that provides intelligent context identification, column relevance mapping, and transpose detection for tabular data. This module uses structured prompting and Pydantic models to extract business context and guide downstream statistical analysis.

## Architecture Decisions

- **Pydantic v2 for structured outputs**: Ensures type-safe responses from LLM calls
- **Structured prompting over direct questions**: Research shows 94-97% accuracy vs 50% for direct "should transpose?" questions
- **Markdown table format for LLM input**: 7% better understanding than raw CSV
- **Two-phase transpose detection**: Header analysis + logic layer separation for testability
- **Async-first design**: All LLM calls are async for better performance

## Technical Approach

### Module Structure
```
src/regression_analyzer/analyzer/
├── __init__.py          # Exports: LLMAnalyzer, models
├── llm_analyzer.py      # Main LLMAnalyzer class
├── prompts.py           # Prompt templates + format_table_as_markdown()
└── models.py            # Pydantic response models
```

### Core Components
1. **Response Models**: BusinessContext, ColumnAnalysis, HeaderAnalysis, TransposeDecision
2. **Prompt Templates**: Context identification, column relevance, header analysis
3. **LLMAnalyzer class**: Orchestrates all analysis with LLMRunner
4. **Logic layer**: Deterministic transpose decision from header analysis

### Integration Points
- Uses LLMRunner from `core/llm_runner.py` (already exists)
- Uses `transpose_dataframe()` from `loader/transpose.py` (PRD-01)
- Outputs Pydantic models for downstream consumers

## Implementation Strategy

The PRD contains complete implementation code. Strategy:
1. Create Pydantic models for structured responses
2. Implement prompt templates with markdown formatting
3. Build LLMAnalyzer class using existing LLMRunner
4. Add comprehensive tests with mocked LLM responses

## Task Breakdown Preview

- [ ] Task 1: Create analyzer module with Pydantic models and prompts
- [ ] Task 2: Implement LLMAnalyzer class with all analysis methods
- [ ] Task 3: Write comprehensive tests for analyzer module

## Dependencies

- `pydantic>=2.0.0` (already in project)
- LLMRunner from `core/` module (already exists)
- DataLoader/transpose from PRD-01 (completed)

## Success Criteria (Technical)

- All Pydantic models validate correctly
- Prompts format tables as markdown
- LLMAnalyzer works with any LLMRunner provider
- Tests pass with mocked LLM responses
- Full analysis pipeline returns structured results

## Estimated Effort

- **Timeline**: 1-2 hours implementation
- **Files to create**: 4 source files + 2 test files
- **Complexity**: Medium (well-specified in PRD, LLMRunner exists)

## Tasks Created
- [ ] #6 - Create analyzer Pydantic models and prompt templates (parallel: true)
- [ ] #7 - Implement LLMAnalyzer class with all analysis methods (parallel: false)
- [ ] #8 - Write comprehensive tests for analyzer module (parallel: false)

Total tasks: 3
Parallel tasks: 1
Sequential tasks: 2
Estimated total effort: 2 hours
