# Regression Analyzer - Claude Code Project

## Overview

Automated tabular data analysis tool with LLM-powered insights, regression analysis, and chart generation.

## Project Structure

```
regression-analyzer/
├── src/regression_analyzer/
│   ├── core/           # LLMRunner (multi-provider)
│   ├── providers/      # LLM provider implementations
│   ├── loader/         # PRD-01: Data loading
│   ├── analyzer/       # PRD-02: LLM analysis
│   ├── stats/          # PRD-03: Statistics engine
│   ├── charts/         # PRD-04: Chart generation
│   ├── cli.py          # PRD-05: CLI interface
│   └── pipeline.py     # PRD-05: Analysis orchestration
├── tests/
├── .claude/prds/       # Implementation specifications
└── pyproject.toml
```

## PRD Implementation Order

1. **PRD-01: data-loader** - CSV/Excel loading, encoding detection, transpose
2. **PRD-02: llm-analyzer** - Context identification, column relevance, transpose detection
3. **PRD-03: stats-engine** - Linear regression, random forest feature importance
4. **PRD-04: chart-generator** - PNG charts for min/max, regression, importance
5. **PRD-05: cli-interface** - Click CLI with Rich output

## Commands

```bash
# Install for development
pip install -e ".[dev,all-providers]"

# Run tests
pytest

# Run analysis
regression-analyzer analyze data.csv --context "company description"
```

## Environment Variables

- `LLM_PROVIDER`: Default provider (claude-code, openai, anthropic, google, groq)
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `GOOGLE_GEMINI_API_KEY`: Google Gemini API key
- `GROQ_API_KEY`: Groq API key

## Key Design Decisions

1. **Polars over Pandas**: 5-10x faster, native sklearn 1.4+ support
2. **Permutation importance**: Unbiased vs tree's `.feature_importances_`
3. **HTML/Markdown for LLM**: 7% better accuracy than CSV format
4. **Structured transpose detection**: 85%+ accuracy vs 50% for direct questions
5. **Seaborn for charts**: Best static quality with minimal code
