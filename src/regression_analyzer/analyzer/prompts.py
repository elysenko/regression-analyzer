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
    sample = df.head(max_rows)
    headers = " | ".join(sample.columns)
    separator = " | ".join(["---"] * len(sample.columns))
    rows = []
    for i in range(sample.height):
        row_vals = [str(sample[col][i]) for col in sample.columns]
        rows.append(" | ".join(row_vals))
    return f"| {headers} |\n| {separator} |\n" + "\n".join(f"| {r} |" for r in rows)
