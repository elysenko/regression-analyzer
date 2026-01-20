# tests/test_pipeline.py

import pytest
import polars as pl
import numpy as np
from pathlib import Path

from regression_analyzer.pipeline import AnalysisPipeline, AnalysisResult


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    csv_file = tmp_path / "test_data.csv"
    np.random.seed(42)
    n = 50

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = 2 * x1 + 3 * x2 + np.random.randn(n) * 0.1

    df = pl.DataFrame({"x1": x1, "x2": x2, "y": y})
    df.write_csv(csv_file)
    return csv_file


class TestAnalysisPipeline:
    """Tests for AnalysisPipeline."""

    @pytest.mark.asyncio
    async def test_load_data(self, sample_csv, tmp_path):
        """Test data loading."""
        pipeline = AnalysisPipeline(
            output_dir=tmp_path / "output",
            use_llm=False
        )

        df = await pipeline.load_data(str(sample_csv))

        assert df is not None
        assert df.height == 50
        assert df.width == 3
        assert "x1" in df.columns

    @pytest.mark.asyncio
    async def test_run_statistics(self, sample_csv, tmp_path):
        """Test statistics execution."""
        pipeline = AnalysisPipeline(
            output_dir=tmp_path / "output",
            use_llm=False
        )

        await pipeline.load_data(str(sample_csv))
        report = await pipeline.run_statistics(target="y", features=["x1", "x2"])

        assert report is not None
        assert report.linear_regression is not None
        assert report.linear_regression.r_squared > 0.9

    @pytest.mark.asyncio
    async def test_generate_charts(self, sample_csv, tmp_path):
        """Test chart generation."""
        pipeline = AnalysisPipeline(
            output_dir=tmp_path / "output",
            use_llm=False
        )

        await pipeline.load_data(str(sample_csv))
        await pipeline.run_statistics(target="y", features=["x1", "x2"])
        charts = await pipeline.generate_charts()

        assert len(charts) > 0
        for chart in charts:
            assert chart.exists()
            assert chart.suffix == ".png"

    @pytest.mark.asyncio
    async def test_get_result(self, sample_csv, tmp_path):
        """Test result aggregation."""
        pipeline = AnalysisPipeline(
            output_dir=tmp_path / "output",
            use_llm=False
        )

        await pipeline.load_data(str(sample_csv))
        await pipeline.run_statistics(target="y")
        await pipeline.generate_charts()

        result = pipeline.get_result()

        assert isinstance(result, AnalysisResult)
        assert result.rows == 50
        assert result.columns == 3
        assert result.statistics is not None

    @pytest.mark.asyncio
    async def test_result_to_dict(self, sample_csv, tmp_path):
        """Test result serialization."""
        pipeline = AnalysisPipeline(
            output_dir=tmp_path / "output",
            use_llm=False
        )

        await pipeline.load_data(str(sample_csv))
        await pipeline.run_statistics(target="y")

        result = pipeline.get_result()
        data = result.to_dict()

        assert isinstance(data, dict)
        assert "file_path" in data
        assert "rows" in data
        assert "statistics" in data

    @pytest.mark.asyncio
    async def test_pipeline_without_llm(self, sample_csv, tmp_path):
        """Test full pipeline without LLM."""
        pipeline = AnalysisPipeline(
            output_dir=tmp_path / "output",
            use_llm=False
        )

        await pipeline.load_data(str(sample_csv))

        # These should be no-ops without LLM
        transpose = await pipeline.check_transpose()
        context = await pipeline.identify_context()
        columns = await pipeline.analyze_columns()

        assert transpose is None
        assert context is None
        assert columns is None

        # Statistics should still work
        await pipeline.run_statistics()
        result = pipeline.get_result()

        assert result.statistics is not None

    @pytest.mark.asyncio
    async def test_output_directory_creation(self, sample_csv, tmp_path):
        """Test that output directory is created."""
        output_dir = tmp_path / "nested" / "output" / "dir"
        assert not output_dir.exists()

        pipeline = AnalysisPipeline(
            output_dir=output_dir,
            use_llm=False
        )

        assert output_dir.exists()


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_result_defaults(self):
        """Test result default values."""
        result = AnalysisResult(
            file_path="test.csv",
            rows=100,
            columns=5
        )

        assert result.transposed is False
        assert result.context is None
        assert result.charts == []
        assert result.warnings == []

    def test_result_to_dict_with_none_values(self):
        """Test serialization with None values."""
        result = AnalysisResult(
            file_path="test.csv",
            rows=100,
            columns=5
        )

        data = result.to_dict()

        assert data["file_path"] == "test.csv"
        assert data["rows"] == 100
        assert data["context"] is None
        assert data["statistics"] is None
