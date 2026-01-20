---
name: llm-analyzer
id: PRD-02
description: LLM-powered context identification, column relevance mapping, and transpose detection
status: backlog
phase: mvp
priority: P0
complexity: high
wave: 2

depends_on:
  - PRD-01  # Needs DataLoader for table serialization

creates:
  - src/regression_analyzer/analyzer/__init__.py
  - src/regression_analyzer/analyzer/llm_analyzer.py
  - src/regression_analyzer/analyzer/prompts.py
  - src/regression_analyzer/analyzer/models.py
  - tests/analyzer/test_llm_analyzer.py
  - tests/analyzer/test_prompts.py

modifies: []

database:
  creates: []
  modifies: []

test_command: pytest tests/analyzer/

blocks: [PRD-03, PRD-05]

references: [PRD-01]

created: 2026-01-20T19:06:34Z
updated: 2026-01-20T19:06:34Z
---

# PRD-02: LLM Analyzer

## Overview

**Feature:** LLM-powered context identification, column relevance mapping, and transpose detection
**Priority:** P0 (Core intelligence layer)
**Complexity:** High
**Dependencies:** PRD-01 (Data Loader)

---

## Problem Statement

Raw tabular data lacks context. The system needs to:
1. Understand what company/domain the data represents
2. Identify which columns relate to key business initiatives (profit, revenue, growth)
3. Detect if the table orientation is inverted (needs transpose)
4. Map columns to their semantic meaning for downstream analysis

Research findings indicate:
- HTML/Markdown table format improves LLM understanding by ~7%
- LLMs achieve 94-97% accuracy on header detection
- Direct "should transpose?" questions fail (~50% accuracy)
- Structured questions with logic application work better

---

## Goals

1. Identify business context from data (company type, industry, focus areas)
2. Map columns to business relevance (high/medium/low impact)
3. Detect transpose need via structured header analysis
4. Return structured Pydantic models for downstream use
5. Support multiple LLM providers via LLMRunner

---

## Non-Goals

- Data transformation (analysis only)
- Real-time streaming analysis
- Multi-table relationship detection
- Domain-specific ontologies

---

## Technical Design

### Architecture

```
analyzer/
├── __init__.py          # Public API exports
├── llm_analyzer.py      # Main LLMAnalyzer class
├── prompts.py           # Prompt templates
└── models.py            # Pydantic response models
```

### Response Models

```python
# src/regression_analyzer/analyzer/models.py

from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class BusinessContext(BaseModel):
    """Identified business context from data."""

    company_type: str = Field(
        description="Type of company (e.g., 'retail', 'SaaS', 'manufacturing')"
    )
    industry: str = Field(
        description="Industry sector (e.g., 'technology', 'finance', 'healthcare')"
    )
    likely_focus_areas: List[str] = Field(
        description="Business metrics they likely care about",
        default_factory=list
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in context identification"
    )


class ColumnRelevance(BaseModel):
    """Relevance assessment for a single column."""

    column_name: str
    semantic_type: str = Field(
        description="What this column represents (e.g., 'revenue', 'date', 'customer_id')"
    )
    relevance: Literal["high", "medium", "low"] = Field(
        description="Relevance to business objectives"
    )
    is_target_candidate: bool = Field(
        description="Whether this could be a prediction target"
    )
    is_feature_candidate: bool = Field(
        description="Whether this could be a useful feature"
    )
    data_quality_notes: Optional[str] = Field(
        default=None,
        description="Any data quality concerns"
    )


class ColumnAnalysis(BaseModel):
    """Full column relevance analysis."""

    columns: List[ColumnRelevance]
    recommended_target: Optional[str] = Field(
        default=None,
        description="Recommended target column for regression"
    )
    recommended_features: List[str] = Field(
        default_factory=list,
        description="Recommended feature columns"
    )


class HeaderAnalysis(BaseModel):
    """Analysis of table headers for transpose detection."""

    first_row_looks_like_headers: bool = Field(
        description="Whether first row appears to contain column headers"
    )
    first_column_looks_like_headers: bool = Field(
        description="Whether first column appears to contain row headers"
    )
    header_examples_row: List[str] = Field(
        description="Examples from first row"
    )
    header_examples_col: List[str] = Field(
        description="Examples from first column"
    )
    reasoning: str = Field(
        description="Explanation of header analysis"
    )


class TransposeDecision(BaseModel):
    """Final transpose recommendation."""

    should_transpose: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
```

### Prompt Templates

```python
# src/regression_analyzer/analyzer/prompts.py

CONTEXT_IDENTIFICATION_PROMPT = '''
Analyze this tabular data to identify the business context.

## Data Sample (Markdown format)
{table_markdown}

## Column Information
{column_info}

Based on the column names, data types, and sample values, identify:
1. What type of company/organization this data likely belongs to
2. What industry sector
3. What business metrics they likely care about (e.g., revenue, profit, churn, engagement)

Respond with JSON:
{{
    "company_type": "...",
    "industry": "...",
    "likely_focus_areas": ["metric1", "metric2", ...],
    "confidence": 0.0-1.0
}}
'''

COLUMN_RELEVANCE_PROMPT = '''
Analyze the relevance of each column for business analysis.

## Business Context
{context}

## Data Sample (Markdown format)
{table_markdown}

## Columns to Analyze
{column_list}

For each column, determine:
1. What it semantically represents
2. Relevance to business objectives (high/medium/low)
3. Whether it could be a prediction target
4. Whether it could be a useful feature
5. Any data quality concerns

Also recommend:
- The best target column for regression (what to predict)
- The best feature columns (what to use for prediction)

Respond with JSON:
{{
    "columns": [
        {{
            "column_name": "...",
            "semantic_type": "...",
            "relevance": "high|medium|low",
            "is_target_candidate": true|false,
            "is_feature_candidate": true|false,
            "data_quality_notes": "..." or null
        }},
        ...
    ],
    "recommended_target": "column_name" or null,
    "recommended_features": ["col1", "col2", ...]
}}
'''

HEADER_ANALYSIS_PROMPT = '''
Analyze this table to determine where the headers are located.

## First 5 Rows (Markdown format)
{table_head_markdown}

## First Column Values (first 10)
{first_column_values}

## First Row Values
{first_row_values}

Analyze:
1. Does the FIRST ROW look like column headers? (short labels, unique, descriptive)
2. Does the FIRST COLUMN look like row headers? (short labels, unique, descriptive)

Consider:
- Headers are typically short text labels
- Headers are usually unique (not repeated)
- Headers describe what the data represents
- Data values are typically numbers, dates, or longer text

Respond with JSON:
{{
    "first_row_looks_like_headers": true|false,
    "first_column_looks_like_headers": true|false,
    "header_examples_row": ["example1", "example2", ...],
    "header_examples_col": ["example1", "example2", ...],
    "reasoning": "explanation of your analysis"
}}
'''


def format_table_as_markdown(df, max_rows: int = 10) -> str:
    """Convert DataFrame to Markdown table format for LLM consumption."""
    # Take sample
    sample = df.head(max_rows)

    # Build markdown
    headers = " | ".join(sample.columns)
    separator = " | ".join(["---"] * len(sample.columns))

    rows = []
    for i in range(sample.height):
        row_vals = [str(sample[col][i]) for col in sample.columns]
        rows.append(" | ".join(row_vals))

    return f"| {headers} |\n| {separator} |\n" + "\n".join(f"| {r} |" for r in rows)
```

### Main Analyzer Class

```python
# src/regression_analyzer/analyzer/llm_analyzer.py

from typing import Optional
import polars as pl

from ..core.llm_runner import LLMRunner
from .models import (
    BusinessContext,
    ColumnAnalysis,
    HeaderAnalysis,
    TransposeDecision,
)
from .prompts import (
    CONTEXT_IDENTIFICATION_PROMPT,
    COLUMN_RELEVANCE_PROMPT,
    HEADER_ANALYSIS_PROMPT,
    format_table_as_markdown,
)


class LLMAnalyzer:
    """LLM-powered data analysis for context and relevance mapping."""

    def __init__(
        self,
        llm_runner: Optional[LLMRunner] = None,
        model: Optional[str] = None
    ):
        """Initialize analyzer.

        Args:
            llm_runner: LLMRunner instance (creates default if None)
            model: Specific model to use (uses provider default if None)
        """
        self.llm = llm_runner or LLMRunner()
        self.model = model

    async def identify_context(
        self,
        df: pl.DataFrame,
        user_context: Optional[str] = None
    ) -> BusinessContext:
        """Identify business context from data.

        Args:
            df: DataFrame to analyze
            user_context: Optional user-provided context hint

        Returns:
            BusinessContext with identified information
        """
        table_md = format_table_as_markdown(df, max_rows=15)

        column_info = "\n".join([
            f"- {col}: {df[col].dtype}, {df[col].null_count()} nulls, "
            f"sample: {df[col].head(3).to_list()}"
            for col in df.columns
        ])

        prompt = CONTEXT_IDENTIFICATION_PROMPT.format(
            table_markdown=table_md,
            column_info=column_info
        )

        if user_context:
            prompt += f"\n\nUser-provided context hint: {user_context}"

        return await self.llm.run_structured(
            prompt=prompt,
            response_model=BusinessContext,
            model=self.model
        )

    async def analyze_columns(
        self,
        df: pl.DataFrame,
        context: BusinessContext
    ) -> ColumnAnalysis:
        """Analyze column relevance for business objectives.

        Args:
            df: DataFrame to analyze
            context: Previously identified business context

        Returns:
            ColumnAnalysis with relevance mapping
        """
        table_md = format_table_as_markdown(df, max_rows=10)

        column_list = "\n".join([
            f"- {col} ({df[col].dtype}): {df[col].head(3).to_list()}"
            for col in df.columns
        ])

        context_str = (
            f"Company type: {context.company_type}\n"
            f"Industry: {context.industry}\n"
            f"Focus areas: {', '.join(context.likely_focus_areas)}"
        )

        prompt = COLUMN_RELEVANCE_PROMPT.format(
            context=context_str,
            table_markdown=table_md,
            column_list=column_list
        )

        return await self.llm.run_structured(
            prompt=prompt,
            response_model=ColumnAnalysis,
            model=self.model
        )

    async def analyze_headers(self, df: pl.DataFrame) -> HeaderAnalysis:
        """Analyze table headers to detect orientation.

        Args:
            df: DataFrame to analyze

        Returns:
            HeaderAnalysis with header location assessment
        """
        table_head_md = format_table_as_markdown(df, max_rows=5)

        first_col_vals = df.select(df.columns[0]).head(10).to_series().to_list()
        first_row_vals = [df[col][0] for col in df.columns]

        prompt = HEADER_ANALYSIS_PROMPT.format(
            table_head_markdown=table_head_md,
            first_column_values=first_col_vals,
            first_row_values=first_row_vals
        )

        return await self.llm.run_structured(
            prompt=prompt,
            response_model=HeaderAnalysis,
            model=self.model
        )

    async def should_transpose(self, df: pl.DataFrame) -> TransposeDecision:
        """Determine if table should be transposed.

        Uses structured header analysis rather than direct question
        (research shows this achieves higher accuracy).

        Args:
            df: DataFrame to analyze

        Returns:
            TransposeDecision with recommendation
        """
        header_analysis = await self.analyze_headers(df)

        # Apply logic based on structured analysis
        # Key insight: if first column looks like headers but first row doesn't,
        # we should transpose
        should_transpose = (
            header_analysis.first_column_looks_like_headers and
            not header_analysis.first_row_looks_like_headers
        )

        # Calculate confidence based on clarity of analysis
        if header_analysis.first_column_looks_like_headers != header_analysis.first_row_looks_like_headers:
            confidence = 0.9  # Clear distinction
        elif not header_analysis.first_column_looks_like_headers and not header_analysis.first_row_looks_like_headers:
            confidence = 0.7  # Neither looks like headers, probably fine
            should_transpose = False
        else:
            confidence = 0.5  # Both look like headers, ambiguous

        reasoning = (
            f"First row as headers: {header_analysis.first_row_looks_like_headers}. "
            f"First column as headers: {header_analysis.first_column_looks_like_headers}. "
            f"{header_analysis.reasoning}"
        )

        return TransposeDecision(
            should_transpose=should_transpose,
            confidence=confidence,
            reasoning=reasoning
        )

    async def full_analysis(
        self,
        df: pl.DataFrame,
        user_context: Optional[str] = None,
        check_transpose: bool = True
    ) -> dict:
        """Run full analysis pipeline.

        Args:
            df: DataFrame to analyze
            user_context: Optional user-provided context
            check_transpose: Whether to check for transpose need

        Returns:
            Dict with context, columns, and optional transpose decision
        """
        result = {}

        # Check transpose first if requested
        if check_transpose:
            transpose_decision = await self.should_transpose(df)
            result["transpose"] = transpose_decision

            if transpose_decision.should_transpose:
                from ..loader.transpose import transpose_dataframe
                df = transpose_dataframe(df)
                result["transposed"] = True

        # Identify context
        context = await self.identify_context(df, user_context)
        result["context"] = context

        # Analyze columns
        columns = await self.analyze_columns(df, context)
        result["columns"] = columns

        return result
```

---

## Implementation Tasks

### Task 1: Response Models
- [ ] Create `models.py` with Pydantic models
- [ ] Define BusinessContext, ColumnRelevance, ColumnAnalysis
- [ ] Define HeaderAnalysis, TransposeDecision
- [ ] Add validation and field descriptions

### Task 2: Prompt Templates
- [ ] Create `prompts.py` with prompt templates
- [ ] Implement `format_table_as_markdown()`
- [ ] Test prompt formatting with sample data

### Task 3: Context Identification
- [ ] Implement `identify_context()` method
- [ ] Handle user-provided context hints
- [ ] Test with various data types

### Task 4: Column Analysis
- [ ] Implement `analyze_columns()` method
- [ ] Map columns to relevance levels
- [ ] Recommend target and features

### Task 5: Transpose Detection
- [ ] Implement `analyze_headers()` method
- [ ] Implement `should_transpose()` with logic layer
- [ ] Test accuracy vs direct questions

### Task 6: Full Pipeline
- [ ] Implement `full_analysis()` orchestrator
- [ ] Handle transpose → reanalysis flow
- [ ] Add error handling and retries

### Task 7: Testing
- [ ] Unit tests for each analysis method
- [ ] Integration tests with LLMRunner
- [ ] Mock LLM responses for CI

---

## Testing Strategy

### Unit Tests
```python
# tests/analyzer/test_llm_analyzer.py

import pytest
from unittest.mock import AsyncMock, MagicMock
import polars as pl

from regression_analyzer.analyzer import LLMAnalyzer
from regression_analyzer.analyzer.models import BusinessContext

@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "date": ["2024-01", "2024-02", "2024-03"],
        "revenue": [100000, 120000, 115000],
        "customers": [500, 550, 540],
        "churn_rate": [0.05, 0.04, 0.045],
    })

@pytest.fixture
def mock_llm_runner():
    runner = MagicMock()
    runner.run_structured = AsyncMock()
    return runner

@pytest.mark.asyncio
async def test_identify_context(sample_df, mock_llm_runner):
    """Test context identification."""
    mock_llm_runner.run_structured.return_value = BusinessContext(
        company_type="SaaS",
        industry="technology",
        likely_focus_areas=["revenue", "churn", "customer_growth"],
        confidence=0.85
    )

    analyzer = LLMAnalyzer(llm_runner=mock_llm_runner)
    context = await analyzer.identify_context(sample_df)

    assert context.company_type == "SaaS"
    assert "revenue" in context.likely_focus_areas

@pytest.mark.asyncio
async def test_transpose_detection_clear_case(mock_llm_runner):
    """Test transpose detection with clear row headers."""
    # DataFrame where first column contains what should be headers
    df = pl.DataFrame({
        "metric": ["revenue", "cost", "profit"],
        "Q1": [100, 60, 40],
        "Q2": [120, 70, 50],
    })

    from regression_analyzer.analyzer.models import HeaderAnalysis

    mock_llm_runner.run_structured.return_value = HeaderAnalysis(
        first_row_looks_like_headers=True,
        first_column_looks_like_headers=True,
        header_examples_row=["metric", "Q1", "Q2"],
        header_examples_col=["revenue", "cost", "profit"],
        reasoning="First column contains metric names that should be column headers"
    )

    analyzer = LLMAnalyzer(llm_runner=mock_llm_runner)
    # This tests the logic layer, not the LLM
```

---

## Acceptance Criteria

1. **Context Identification**: Correctly identifies company type for 80%+ of datasets
2. **Column Relevance**: Maps all columns with appropriate relevance levels
3. **Target Recommendation**: Recommends sensible target column for regression
4. **Transpose Detection**: Achieves 85%+ accuracy using structured approach
5. **Provider Agnostic**: Works with all LLMRunner providers
6. **Structured Output**: Returns valid Pydantic models

---

## Dependencies

- `pydantic>=2.0.0` - Response model validation
- LLMRunner from `core/` module
- DataLoader from PRD-01 (for transpose function)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| LLM returns invalid JSON | Use instructor for providers that support it, fallback parsing |
| Transpose detection ambiguous | Return confidence score, let user override |
| Context identification wrong | Allow user-provided context to override/supplement |
| Rate limiting | Batch calls where possible, add retry logic |
