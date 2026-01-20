"""Tests for analyzer models."""

import pytest
from pydantic import ValidationError

from regression_analyzer.analyzer.models import (
    BusinessContext,
    ColumnRelevance,
    ColumnAnalysis,
    HeaderAnalysis,
    TransposeDecision,
)


class TestBusinessContext:
    """Tests for BusinessContext model."""

    def test_valid_business_context(self):
        """Test creating valid BusinessContext."""
        context = BusinessContext(
            company_type="SaaS",
            industry="technology",
            likely_focus_areas=["revenue", "churn", "customer_growth"],
            confidence=0.85
        )
        assert context.company_type == "SaaS"
        assert context.industry == "technology"
        assert "revenue" in context.likely_focus_areas
        assert context.confidence == 0.85

    def test_confidence_at_boundaries(self):
        """Test confidence at valid boundary values."""
        # Test 0.0
        context_min = BusinessContext(
            company_type="retail",
            industry="commerce",
            likely_focus_areas=[],
            confidence=0.0
        )
        assert context_min.confidence == 0.0

        # Test 1.0
        context_max = BusinessContext(
            company_type="retail",
            industry="commerce",
            likely_focus_areas=[],
            confidence=1.0
        )
        assert context_max.confidence == 1.0

    def test_confidence_below_zero_invalid(self):
        """Test that confidence below 0.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BusinessContext(
                company_type="retail",
                industry="commerce",
                likely_focus_areas=[],
                confidence=-0.1
            )
        assert "confidence" in str(exc_info.value).lower()

    def test_confidence_above_one_invalid(self):
        """Test that confidence above 1.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BusinessContext(
                company_type="retail",
                industry="commerce",
                likely_focus_areas=[],
                confidence=1.1
            )
        assert "confidence" in str(exc_info.value).lower()

    def test_empty_focus_areas(self):
        """Test BusinessContext with empty focus areas list."""
        context = BusinessContext(
            company_type="startup",
            industry="fintech",
            likely_focus_areas=[],
            confidence=0.5
        )
        assert context.likely_focus_areas == []

    def test_default_focus_areas(self):
        """Test that likely_focus_areas defaults to empty list."""
        context = BusinessContext(
            company_type="startup",
            industry="fintech",
            confidence=0.5
        )
        assert context.likely_focus_areas == []


class TestColumnRelevance:
    """Tests for ColumnRelevance model."""

    def test_valid_column_relevance_high(self):
        """Test creating ColumnRelevance with high relevance."""
        col = ColumnRelevance(
            column_name="revenue",
            semantic_type="monetary",
            relevance="high",
            is_target_candidate=True,
            is_feature_candidate=False
        )
        assert col.column_name == "revenue"
        assert col.relevance == "high"
        assert col.is_target_candidate is True
        assert col.is_feature_candidate is False

    def test_valid_column_relevance_medium(self):
        """Test creating ColumnRelevance with medium relevance."""
        col = ColumnRelevance(
            column_name="customer_count",
            semantic_type="count",
            relevance="medium",
            is_target_candidate=False,
            is_feature_candidate=True
        )
        assert col.relevance == "medium"

    def test_valid_column_relevance_low(self):
        """Test creating ColumnRelevance with low relevance."""
        col = ColumnRelevance(
            column_name="internal_id",
            semantic_type="identifier",
            relevance="low",
            is_target_candidate=False,
            is_feature_candidate=False
        )
        assert col.relevance == "low"

    def test_invalid_relevance_value(self):
        """Test that invalid relevance raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ColumnRelevance(
                column_name="test",
                semantic_type="test",
                relevance="invalid",
                is_target_candidate=False,
                is_feature_candidate=False
            )
        assert "relevance" in str(exc_info.value).lower()

    def test_data_quality_notes_optional(self):
        """Test that data_quality_notes is optional."""
        col = ColumnRelevance(
            column_name="revenue",
            semantic_type="monetary",
            relevance="high",
            is_target_candidate=True,
            is_feature_candidate=False
        )
        assert col.data_quality_notes is None

    def test_data_quality_notes_present(self):
        """Test ColumnRelevance with data_quality_notes."""
        col = ColumnRelevance(
            column_name="revenue",
            semantic_type="monetary",
            relevance="high",
            is_target_candidate=True,
            is_feature_candidate=False,
            data_quality_notes="Contains some null values"
        )
        assert col.data_quality_notes == "Contains some null values"


class TestColumnAnalysis:
    """Tests for ColumnAnalysis model."""

    def test_valid_column_analysis(self):
        """Test creating valid ColumnAnalysis."""
        col1 = ColumnRelevance(
            column_name="revenue",
            semantic_type="monetary",
            relevance="high",
            is_target_candidate=True,
            is_feature_candidate=False
        )
        col2 = ColumnRelevance(
            column_name="date",
            semantic_type="temporal",
            relevance="medium",
            is_target_candidate=False,
            is_feature_candidate=True
        )

        analysis = ColumnAnalysis(
            columns=[col1, col2],
            recommended_target="revenue",
            recommended_features=["date"]
        )

        assert len(analysis.columns) == 2
        assert analysis.recommended_target == "revenue"
        assert "date" in analysis.recommended_features

    def test_empty_columns_list(self):
        """Test ColumnAnalysis with empty columns list."""
        analysis = ColumnAnalysis(
            columns=[],
            recommended_target=None,
            recommended_features=[]
        )
        assert analysis.columns == []

    def test_recommended_target_optional(self):
        """Test that recommended_target is optional."""
        col = ColumnRelevance(
            column_name="test",
            semantic_type="test",
            relevance="low",
            is_target_candidate=False,
            is_feature_candidate=False
        )
        analysis = ColumnAnalysis(columns=[col])
        assert analysis.recommended_target is None

    def test_default_recommended_features(self):
        """Test that recommended_features defaults to empty list."""
        col = ColumnRelevance(
            column_name="test",
            semantic_type="test",
            relevance="low",
            is_target_candidate=False,
            is_feature_candidate=False
        )
        analysis = ColumnAnalysis(columns=[col])
        assert analysis.recommended_features == []


class TestHeaderAnalysis:
    """Tests for HeaderAnalysis model."""

    def test_valid_header_analysis(self):
        """Test creating valid HeaderAnalysis."""
        analysis = HeaderAnalysis(
            first_row_looks_like_headers=True,
            first_column_looks_like_headers=False,
            header_examples_row=["date", "revenue", "customers"],
            header_examples_col=["2024-01", "2024-02"],
            reasoning="First row contains column names"
        )

        assert analysis.first_row_looks_like_headers is True
        assert analysis.first_column_looks_like_headers is False
        assert len(analysis.header_examples_row) == 3
        assert len(analysis.header_examples_col) == 2

    def test_both_headers_true(self):
        """Test HeaderAnalysis where both dimensions look like headers."""
        analysis = HeaderAnalysis(
            first_row_looks_like_headers=True,
            first_column_looks_like_headers=True,
            header_examples_row=["A", "B", "C"],
            header_examples_col=["X", "Y", "Z"],
            reasoning="Ambiguous - both look like headers"
        )
        assert analysis.first_row_looks_like_headers
        assert analysis.first_column_looks_like_headers

    def test_neither_headers_true(self):
        """Test HeaderAnalysis where neither dimension looks like headers."""
        analysis = HeaderAnalysis(
            first_row_looks_like_headers=False,
            first_column_looks_like_headers=False,
            header_examples_row=["1", "2", "3"],
            header_examples_col=["10", "20", "30"],
            reasoning="All values appear to be data"
        )
        assert not analysis.first_row_looks_like_headers
        assert not analysis.first_column_looks_like_headers

    def test_empty_example_lists(self):
        """Test HeaderAnalysis with empty example lists."""
        analysis = HeaderAnalysis(
            first_row_looks_like_headers=True,
            first_column_looks_like_headers=False,
            header_examples_row=[],
            header_examples_col=[],
            reasoning="Empty table"
        )
        assert analysis.header_examples_row == []
        assert analysis.header_examples_col == []


class TestTransposeDecision:
    """Tests for TransposeDecision model."""

    def test_valid_transpose_decision_true(self):
        """Test TransposeDecision recommending transpose."""
        decision = TransposeDecision(
            should_transpose=True,
            confidence=0.9,
            reasoning="First column contains metric names"
        )
        assert decision.should_transpose is True
        assert decision.confidence == 0.9

    def test_valid_transpose_decision_false(self):
        """Test TransposeDecision not recommending transpose."""
        decision = TransposeDecision(
            should_transpose=False,
            confidence=0.85,
            reasoning="Standard orientation detected"
        )
        assert decision.should_transpose is False
        assert decision.confidence == 0.85

    def test_confidence_at_boundaries(self):
        """Test confidence at valid boundary values."""
        # Test 0.0
        decision_min = TransposeDecision(
            should_transpose=False,
            confidence=0.0,
            reasoning="Cannot determine"
        )
        assert decision_min.confidence == 0.0

        # Test 1.0
        decision_max = TransposeDecision(
            should_transpose=True,
            confidence=1.0,
            reasoning="Definitely needs transpose"
        )
        assert decision_max.confidence == 1.0

    def test_confidence_below_zero_invalid(self):
        """Test that confidence below 0.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TransposeDecision(
                should_transpose=False,
                confidence=-0.1,
                reasoning="Test"
            )
        assert "confidence" in str(exc_info.value).lower()

    def test_confidence_above_one_invalid(self):
        """Test that confidence above 1.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TransposeDecision(
                should_transpose=False,
                confidence=1.1,
                reasoning="Test"
            )
        assert "confidence" in str(exc_info.value).lower()
