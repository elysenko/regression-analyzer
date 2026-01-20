"""LLM-powered data analyzer for context and relevance mapping."""

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
