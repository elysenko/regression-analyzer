# Regression Analyzer

Automated tabular data analysis with LLM-powered insights, regression analysis, and chart generation.

## Features

- **Smart Data Loading**: CSV/Excel with automatic encoding detection
- **LLM-Powered Analysis**: Context identification, column relevance mapping, transpose detection
- **Statistical Analysis**: Linear regression and Random Forest feature importance
- **Chart Generation**: Min/max visualization, regression plots (PNG output)
- **Multi-Provider LLM**: OpenAI, Anthropic, Google, Groq, or Claude Code CLI

## Installation

```bash
pip install regression-analyzer

# With specific LLM providers
pip install regression-analyzer[openai]
pip install regression-analyzer[anthropic]
pip install regression-analyzer[all-providers]
```

## Quick Start

```bash
# Set your preferred LLM provider
export LLM_PROVIDER=claude-code  # or openai, anthropic, google, groq
export OPENAI_API_KEY=sk-...     # if using OpenAI

# Analyze a dataset
regression-analyzer analyze data.csv --context "retail company focused on profit margins"
```

## Usage

```bash
# Basic analysis
regression-analyzer analyze sales_data.csv

# With business context
regression-analyzer analyze financials.xlsx \
  --context "SaaS company, care about MRR and churn" \
  --output-dir ./results

# Transpose detection
regression-analyzer analyze wide_table.csv --check-transpose

# Specific target column
regression-analyzer analyze data.csv --target revenue
```

## Development

See `.claude/prds/` for implementation specifications.

```bash
# Install for development
pip install -e ".[dev,all-providers]"

# Run tests
pytest
```

## License

MIT
