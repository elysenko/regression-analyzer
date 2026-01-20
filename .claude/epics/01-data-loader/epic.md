---
name: 01-data-loader
status: backlog
created: 2026-01-20T19:16:22Z
progress: 0%
prd: .claude/prds/01-data-loader.md
github: https://github.com/elysenko/regression-analyzer/issues/1
---

# Epic: 01-data-loader

## Overview

Implement a robust data loading module that handles CSV and Excel files with automatic encoding detection and transpose capabilities. This is the foundation module that all other PRDs depend on.

## Architecture Decisions

- **Polars as primary DataFrame library**: 5-10x faster than pandas, better memory efficiency
- **Pandas bridge for Excel**: openpyxl requires pandas, convert to Polars after loading
- **Chardet for encoding detection**: Industry standard, handles edge cases well
- **Fallback encoding chain**: Graceful degradation when detection fails

## Technical Approach

### Module Structure
```
src/regression_analyzer/loader/
├── __init__.py          # Exports: DataLoader, transpose_dataframe, detect_encoding
├── data_loader.py       # Main DataLoader class
├── encoding.py          # detect_encoding() function
└── transpose.py         # transpose_dataframe(), should_transpose_heuristic()
```

### Core Components
1. **DataLoader class**: Unified interface for CSV/Excel loading
2. **Encoding detection**: chardet-based with fallback chain
3. **Transpose utility**: Convert row-oriented to column-oriented data
4. **Metadata extraction**: Shape, dtypes, nulls, samples

## Implementation Strategy

The PRD already contains complete implementation code. Strategy:
1. Create module structure
2. Implement code from PRD specifications
3. Add comprehensive tests
4. Validate with real-world data files

## Task Breakdown Preview

- [ ] Task 1: Create loader module with DataLoader class and encoding detection
- [ ] Task 2: Add transpose functionality and heuristics
- [ ] Task 3: Write comprehensive tests for all components

## Dependencies

- `polars>=0.20.0`
- `pandas>=2.0.0` (for Excel bridge)
- `openpyxl>=3.1.0`
- `chardet>=5.0.0`

## Success Criteria (Technical)

- CSV loading works with UTF-8, Latin-1, Windows-1252 encodings
- Excel .xlsx files load correctly (first sheet)
- Transpose function handles duplicate headers
- get_metadata() returns accurate information
- All tests pass
- 100K row CSV loads in < 2 seconds

## Estimated Effort

- **Timeline**: 1-2 hours implementation
- **Files to create**: 4 source files + 3 test files
- **Complexity**: Medium (well-specified in PRD)

## Tasks Created

- [ ] 001.md - Implement DataLoader class with CSV/Excel support and encoding detection (parallel: true)
- [ ] 002.md - Implement transpose functionality (parallel: true)
- [ ] 003.md - Write comprehensive tests for loader module (parallel: false)

Total tasks: 3
Parallel tasks: 2
Sequential tasks: 1
Estimated total effort: 2.5 hours
