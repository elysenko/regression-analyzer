"""Tests for LLMAnalyzer class."""

import pytest
from unittest.mock import AsyncMock, MagicMock
import polars as pl

from regression_analyzer.analyzer import LLMAnalyzer
from regression_analyzer.analyzer.models import (
    BusinessContext,
    ColumnAnalysis,
    ColumnRelevance,
    HeaderAnalysis,
    TransposeDecision,
)


@pytest.fixture
def sample_df():
    """Sample DataFrame for testing."""
    return pl.DataFrame({
        "date": ["2024-01", "2024-02", "2024-03"],
        "revenue": [100000, 120000, 115000],
        "customers": [500, 550, 540],
        "churn_rate": [0.05, 0.04, 0.045],
    })


@pytest.fixture
def mock_llm_runner():
    """Mock LLMRunner for testing."""
    runner = MagicMock()
    runner.run_structured = AsyncMock()
    return runner


class TestLLMAnalyzerInit:
    """Tests for LLMAnalyzer initialization."""

    def test_init_with_llm_runner(self, mock_llm_runner):
        """Test initialization with provided LLMRunner."""
        analyzer = LLMAnalyzer(llm_runner=mock_llm_runner)
        assert analyzer.llm is mock_llm_runner
        assert analyzer.model is None

    def test_init_with_model(self, mock_llm_runner):
        """Test initialization with specific model."""
        analyzer = LLMAnalyzer(llm_runner=mock_llm_runner, model="gpt-4")
        assert analyzer.model == "gpt-4"


class TestIdentifyContext:
    """Tests for identify_context method."""

    @pytest.mark.asyncio
    async def test_identify_context(self, sample_df, mock_llm_runner):
        """Test basic context identification."""
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
        mock_llm_runner.run_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_identify_context_with_user_hint(self, sample_df, mock_llm_runner):
        """Test context identification with user-provided context."""
        mock_llm_runner.run_structured.return_value = BusinessContext(
            company_type="E-commerce",
            industry="retail",
            likely_focus_areas=["sales", "conversion"],
            confidence=0.9
        )

        analyzer = LLMAnalyzer(llm_runner=mock_llm_runner)
        await analyzer.identify_context(sample_df, user_context="This is an e-commerce company")

        # Verify the prompt included the user context
        call_args = mock_llm_runner.run_structured.call_args
        assert "e-commerce" in call_args.kwargs["prompt"].lower()

    @pytest.mark.asyncio
    async def test_identify_context_passes_response_model(self, sample_df, mock_llm_runner):
        """Test that identify_context passes correct response model."""
        mock_llm_runner.run_structured.return_value = BusinessContext(
            company_type="Test",
            industry="test",
            likely_focus_areas=[],
            confidence=0.5
        )

        analyzer = LLMAnalyzer(llm_runner=mock_llm_runner)
        await analyzer.identify_context(sample_df)

        call_args = mock_llm_runner.run_structured.call_args
        assert call_args.kwargs["response_model"] == BusinessContext

    @pytest.mark.asyncio
    async def test_identify_context_uses_model_param(self, sample_df, mock_llm_runner):
        """Test that identify_context uses specified model."""
        mock_llm_runner.run_structured.return_value = BusinessContext(
            company_type="Test",
            industry="test",
            likely_focus_areas=[],
            confidence=0.5
        )

        analyzer = LLMAnalyzer(llm_runner=mock_llm_runner, model="custom-model")
        await analyzer.identify_context(sample_df)

        call_args = mock_llm_runner.run_structured.call_args
        assert call_args.kwargs["model"] == "custom-model"


class TestAnalyzeColumns:
    """Tests for analyze_columns method."""

    @pytest.mark.asyncio
    async def test_analyze_columns(self, sample_df, mock_llm_runner):
        """Test basic column analysis."""
        context = BusinessContext(
            company_type="SaaS",
            industry="technology",
            likely_focus_areas=["revenue"],
            confidence=0.8
        )

        mock_llm_runner.run_structured.return_value = ColumnAnalysis(
            columns=[
                ColumnRelevance(
                    column_name="revenue",
                    semantic_type="monetary",
                    relevance="high",
                    is_target_candidate=True,
                    is_feature_candidate=False
                )
            ],
            recommended_target="revenue",
            recommended_features=["customers", "date"]
        )

        analyzer = LLMAnalyzer(llm_runner=mock_llm_runner)
        analysis = await analyzer.analyze_columns(sample_df, context)

        assert analysis.recommended_target == "revenue"
        assert len(analysis.columns) == 1

    @pytest.mark.asyncio
    async def test_analyze_columns_passes_context_in_prompt(self, sample_df, mock_llm_runner):
        """Test that analyze_columns includes context in prompt."""
        context = BusinessContext(
            company_type="Healthcare",
            industry="medical",
            likely_focus_areas=["patient_outcomes"],
            confidence=0.8
        )

        mock_llm_runner.run_structured.return_value = ColumnAnalysis(
            columns=[],
            recommended_target=None,
            recommended_features=[]
        )

        analyzer = LLMAnalyzer(llm_runner=mock_llm_runner)
        await analyzer.analyze_columns(sample_df, context)

        call_args = mock_llm_runner.run_structured.call_args
        prompt = call_args.kwargs["prompt"]
        assert "Healthcare" in prompt
        assert "medical" in prompt

    @pytest.mark.asyncio
    async def test_analyze_columns_passes_response_model(self, sample_df, mock_llm_runner):
        """Test that analyze_columns passes correct response model."""
        context = BusinessContext(
            company_type="Test",
            industry="test",
            likely_focus_areas=[],
            confidence=0.5
        )

        mock_llm_runner.run_structured.return_value = ColumnAnalysis(
            columns=[],
            recommended_target=None,
            recommended_features=[]
        )

        analyzer = LLMAnalyzer(llm_runner=mock_llm_runner)
        await analyzer.analyze_columns(sample_df, context)

        call_args = mock_llm_runner.run_structured.call_args
        assert call_args.kwargs["response_model"] == ColumnAnalysis


class TestAnalyzeHeaders:
    """Tests for analyze_headers method."""

    @pytest.mark.asyncio
    async def test_analyze_headers(self, sample_df, mock_llm_runner):
        """Test basic header analysis."""
        mock_llm_runner.run_structured.return_value = HeaderAnalysis(
            first_row_looks_like_headers=True,
            first_column_looks_like_headers=False,
            header_examples_row=["date", "revenue", "customers"],
            header_examples_col=["2024-01", "2024-02"],
            reasoning="First row contains column names"
        )

        analyzer = LLMAnalyzer(llm_runner=mock_llm_runner)
        analysis = await analyzer.analyze_headers(sample_df)

        assert analysis.first_row_looks_like_headers
        assert not analysis.first_column_looks_like_headers

    @pytest.mark.asyncio
    async def test_analyze_headers_passes_response_model(self, sample_df, mock_llm_runner):
        """Test that analyze_headers passes correct response model."""
        mock_llm_runner.run_structured.return_value = HeaderAnalysis(
            first_row_looks_like_headers=True,
            first_column_looks_like_headers=False,
            header_examples_row=[],
            header_examples_col=[],
            reasoning="Test"
        )

        analyzer = LLMAnalyzer(llm_runner=mock_llm_runner)
        await analyzer.analyze_headers(sample_df)

        call_args = mock_llm_runner.run_structured.call_args
        assert call_args.kwargs["response_model"] == HeaderAnalysis


class TestShouldTranspose:
    """Tests for should_transpose method."""

    @pytest.mark.asyncio
    async def test_should_transpose_no_transpose_needed(self, sample_df, mock_llm_runner):
        """Test when no transpose is needed (standard orientation)."""
        mock_llm_runner.run_structured.return_value = HeaderAnalysis(
            first_row_looks_like_headers=True,
            first_column_looks_like_headers=False,
            header_examples_row=["date", "revenue"],
            header_examples_col=["2024-01"],
            reasoning="Standard orientation"
        )

        analyzer = LLMAnalyzer(llm_runner=mock_llm_runner)
        decision = await analyzer.should_transpose(sample_df)

        assert not decision.should_transpose
        assert decision.confidence == 0.9

    @pytest.mark.asyncio
    async def test_should_transpose_transpose_needed(self):
        """Test when transpose is needed (first column has headers)."""
        df = pl.DataFrame({
            "metric": ["revenue", "cost", "profit"],
            "Q1": [100, 60, 40],
            "Q2": [120, 70, 50],
        })

        mock_runner = MagicMock()
        mock_runner.run_structured = AsyncMock(return_value=HeaderAnalysis(
            first_row_looks_like_headers=False,
            first_column_looks_like_headers=True,
            header_examples_row=["metric", "Q1", "Q2"],
            header_examples_col=["revenue", "cost", "profit"],
            reasoning="First column contains metric names"
        ))

        analyzer = LLMAnalyzer(llm_runner=mock_runner)
        decision = await analyzer.should_transpose(df)

        assert decision.should_transpose
        assert decision.confidence == 0.9

    @pytest.mark.asyncio
    async def test_should_transpose_ambiguous(self):
        """Test ambiguous case where both look like headers."""
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

        mock_runner = MagicMock()
        mock_runner.run_structured = AsyncMock(return_value=HeaderAnalysis(
            first_row_looks_like_headers=True,
            first_column_looks_like_headers=True,
            header_examples_row=["a", "b"],
            header_examples_col=["1", "2"],
            reasoning="Ambiguous"
        ))

        analyzer = LLMAnalyzer(llm_runner=mock_runner)
        decision = await analyzer.should_transpose(df)

        assert decision.confidence == 0.5  # Ambiguous case

    @pytest.mark.asyncio
    async def test_should_transpose_neither_looks_like_headers(self):
        """Test when neither row nor column looks like headers."""
        df = pl.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        mock_runner = MagicMock()
        mock_runner.run_structured = AsyncMock(return_value=HeaderAnalysis(
            first_row_looks_like_headers=False,
            first_column_looks_like_headers=False,
            header_examples_row=["col1", "col2"],
            header_examples_col=["1", "2"],
            reasoning="All values appear to be data"
        ))

        analyzer = LLMAnalyzer(llm_runner=mock_runner)
        decision = await analyzer.should_transpose(df)

        assert not decision.should_transpose
        assert decision.confidence == 0.7

    @pytest.mark.asyncio
    async def test_should_transpose_includes_reasoning(self, sample_df, mock_llm_runner):
        """Test that transpose decision includes reasoning from header analysis."""
        mock_llm_runner.run_structured.return_value = HeaderAnalysis(
            first_row_looks_like_headers=True,
            first_column_looks_like_headers=False,
            header_examples_row=["date", "revenue"],
            header_examples_col=["2024-01"],
            reasoning="Column names are descriptive text"
        )

        analyzer = LLMAnalyzer(llm_runner=mock_llm_runner)
        decision = await analyzer.should_transpose(sample_df)

        assert "Column names are descriptive text" in decision.reasoning


class TestFullAnalysis:
    """Tests for full_analysis method."""

    @pytest.mark.asyncio
    async def test_full_analysis_without_transpose_check(self, sample_df, mock_llm_runner):
        """Test full analysis with transpose check disabled."""
        mock_llm_runner.run_structured.side_effect = [
            BusinessContext(
                company_type="SaaS",
                industry="technology",
                likely_focus_areas=["revenue"],
                confidence=0.8
            ),
            ColumnAnalysis(
                columns=[],
                recommended_target="revenue",
                recommended_features=["customers"]
            )
        ]

        analyzer = LLMAnalyzer(llm_runner=mock_llm_runner)
        result = await analyzer.full_analysis(sample_df, check_transpose=False)

        assert "context" in result
        assert "columns" in result
        assert "transpose" not in result
        assert result["context"].company_type == "SaaS"

    @pytest.mark.asyncio
    async def test_full_analysis_with_transpose_check(self, sample_df, mock_llm_runner):
        """Test full analysis with transpose check enabled."""
        mock_llm_runner.run_structured.side_effect = [
            HeaderAnalysis(
                first_row_looks_like_headers=True,
                first_column_looks_like_headers=False,
                header_examples_row=["date", "revenue"],
                header_examples_col=["2024-01"],
                reasoning="Standard orientation"
            ),
            BusinessContext(
                company_type="SaaS",
                industry="technology",
                likely_focus_areas=["revenue"],
                confidence=0.8
            ),
            ColumnAnalysis(
                columns=[],
                recommended_target="revenue",
                recommended_features=["customers"]
            )
        ]

        analyzer = LLMAnalyzer(llm_runner=mock_llm_runner)
        result = await analyzer.full_analysis(sample_df, check_transpose=True)

        assert "context" in result
        assert "columns" in result
        assert "transpose" in result
        assert not result["transpose"].should_transpose

    @pytest.mark.asyncio
    async def test_full_analysis_with_user_context(self, sample_df, mock_llm_runner):
        """Test full analysis with user-provided context."""
        mock_llm_runner.run_structured.side_effect = [
            BusinessContext(
                company_type="E-commerce",
                industry="retail",
                likely_focus_areas=["sales"],
                confidence=0.9
            ),
            ColumnAnalysis(
                columns=[],
                recommended_target="revenue",
                recommended_features=[]
            )
        ]

        analyzer = LLMAnalyzer(llm_runner=mock_llm_runner)
        result = await analyzer.full_analysis(
            sample_df,
            user_context="This is an e-commerce store",
            check_transpose=False
        )

        # Verify user context was passed
        first_call_args = mock_llm_runner.run_structured.call_args_list[0]
        assert "e-commerce" in first_call_args.kwargs["prompt"].lower()


class TestLLMAnalyzerEdgeCases:
    """Edge case tests for LLMAnalyzer."""

    @pytest.mark.asyncio
    async def test_analyze_single_column_df(self, mock_llm_runner):
        """Test analysis of single-column DataFrame."""
        df = pl.DataFrame({"only_column": [1, 2, 3]})

        mock_llm_runner.run_structured.return_value = BusinessContext(
            company_type="unknown",
            industry="unknown",
            likely_focus_areas=[],
            confidence=0.3
        )

        analyzer = LLMAnalyzer(llm_runner=mock_llm_runner)
        context = await analyzer.identify_context(df)

        assert context.confidence == 0.3

    @pytest.mark.asyncio
    async def test_analyze_single_row_df(self, mock_llm_runner):
        """Test analysis of single-row DataFrame."""
        df = pl.DataFrame({
            "col1": [100],
            "col2": [200],
            "col3": [300]
        })

        mock_llm_runner.run_structured.return_value = HeaderAnalysis(
            first_row_looks_like_headers=False,
            first_column_looks_like_headers=False,
            header_examples_row=["col1", "col2", "col3"],
            header_examples_col=["100"],
            reasoning="Only one row of data"
        )

        analyzer = LLMAnalyzer(llm_runner=mock_llm_runner)
        analysis = await analyzer.analyze_headers(df)

        assert not analysis.first_row_looks_like_headers

    @pytest.mark.asyncio
    async def test_analyze_df_with_null_values(self, mock_llm_runner):
        """Test analysis of DataFrame containing nulls."""
        df = pl.DataFrame({
            "name": ["Alice", None, "Charlie"],
            "value": [100, 200, None]
        })

        mock_llm_runner.run_structured.return_value = BusinessContext(
            company_type="generic",
            industry="unknown",
            likely_focus_areas=[],
            confidence=0.5
        )

        analyzer = LLMAnalyzer(llm_runner=mock_llm_runner)
        context = await analyzer.identify_context(df)

        # Should complete without error
        assert context is not None
