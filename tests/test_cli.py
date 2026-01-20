# tests/test_cli.py

import pytest
from click.testing import CliRunner
import json
import numpy as np
import polars as pl

from regression_analyzer.cli import main


@pytest.fixture
def runner():
    """CLI test runner."""
    return CliRunner()


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


class TestAnalyzeCommand:
    """Tests for the analyze command."""

    def test_analyze_basic(self, runner, sample_csv, tmp_path):
        """Test basic analyze command."""
        result = runner.invoke(main, [
            'analyze', str(sample_csv),
            '--output-dir', str(tmp_path / 'output'),
            '--skip-llm'
        ])

        assert result.exit_code == 0
        assert "Analysis complete" in result.output

    def test_analyze_with_target(self, runner, sample_csv, tmp_path):
        """Test analyze with explicit target."""
        result = runner.invoke(main, [
            'analyze', str(sample_csv),
            '--output-dir', str(tmp_path / 'output'),
            '--skip-llm',
            '--target', 'y'
        ])

        assert result.exit_code == 0
        assert "R²" in result.output or "Analysis complete" in result.output

    def test_analyze_json_output(self, runner, sample_csv, tmp_path):
        """Test JSON output mode."""
        result = runner.invoke(main, [
            'analyze', str(sample_csv),
            '--output-dir', str(tmp_path / 'output'),
            '--skip-llm',
            '--json-output'
        ])

        assert result.exit_code == 0

        # Parse JSON output
        data = json.loads(result.output)
        assert "file_path" in data
        assert "rows" in data
        assert "statistics" in data
        assert data["rows"] == 50

    def test_analyze_creates_output_dir(self, runner, sample_csv, tmp_path):
        """Test that output directory is created."""
        output_dir = tmp_path / "new_output_dir"
        assert not output_dir.exists()

        result = runner.invoke(main, [
            'analyze', str(sample_csv),
            '--output-dir', str(output_dir),
            '--skip-llm'
        ])

        assert result.exit_code == 0
        assert output_dir.exists()

    def test_analyze_generates_charts(self, runner, sample_csv, tmp_path):
        """Test that charts are generated."""
        output_dir = tmp_path / "output"

        result = runner.invoke(main, [
            'analyze', str(sample_csv),
            '--output-dir', str(output_dir),
            '--skip-llm'
        ])

        assert result.exit_code == 0

        # Check charts directory
        charts_dir = output_dir / "charts"
        assert charts_dir.exists()
        png_files = list(charts_dir.glob("*.png"))
        assert len(png_files) > 0

    def test_analyze_no_transpose_flag(self, runner, sample_csv, tmp_path):
        """Test no-transpose flag."""
        result = runner.invoke(main, [
            'analyze', str(sample_csv),
            '--output-dir', str(tmp_path / 'output'),
            '--skip-llm',
            '--no-transpose'
        ])

        assert result.exit_code == 0

    def test_analyze_file_not_found(self, runner, tmp_path):
        """Test error handling for missing file."""
        result = runner.invoke(main, [
            'analyze', str(tmp_path / 'nonexistent.csv'),
            '--skip-llm'
        ])

        assert result.exit_code != 0


class TestProvidersCommand:
    """Tests for the providers command."""

    def test_providers_command(self, runner):
        """Test providers listing."""
        result = runner.invoke(main, ['providers'])

        # Should succeed even without API keys
        assert result.exit_code == 0
        assert "Available" in result.output or "Error" in result.output


class TestMainGroup:
    """Tests for the main CLI group."""

    def test_version_option(self, runner):
        """Test version flag."""
        result = runner.invoke(main, ['--version'])

        # Version option may fail if package not installed in dev mode
        # Either exit 0 with version or exception about not being installed
        if result.exit_code != 0:
            assert result.exception is not None
            assert "not installed" in str(result.exception)
        else:
            assert "version" in result.output.lower() or "0." in result.output

    def test_help_option(self, runner):
        """Test help flag."""
        result = runner.invoke(main, ['--help'])

        assert result.exit_code == 0
        assert "Regression Analyzer" in result.output
        assert "analyze" in result.output
        assert "providers" in result.output

    def test_analyze_help(self, runner):
        """Test analyze command help."""
        result = runner.invoke(main, ['analyze', '--help'])

        assert result.exit_code == 0
        assert "--context" in result.output
        assert "--target" in result.output
        assert "--output-dir" in result.output
        assert "--skip-llm" in result.output


class TestJsonOutput:
    """Tests for JSON output format."""

    def test_json_structure(self, runner, sample_csv, tmp_path):
        """Test JSON output structure."""
        result = runner.invoke(main, [
            'analyze', str(sample_csv),
            '--output-dir', str(tmp_path / 'output'),
            '--skip-llm',
            '-j'
        ])

        data = json.loads(result.output)

        # Check required fields
        assert "file_path" in data
        assert "rows" in data
        assert "columns" in data
        assert "transposed" in data
        assert "statistics" in data
        assert "charts" in data
        assert "warnings" in data

    def test_json_statistics_content(self, runner, sample_csv, tmp_path):
        """Test statistics content in JSON."""
        result = runner.invoke(main, [
            'analyze', str(sample_csv),
            '--output-dir', str(tmp_path / 'output'),
            '--skip-llm',
            '--target', 'y',
            '-j'
        ])

        data = json.loads(result.output)
        stats = data["statistics"]

        assert stats is not None
        assert "linear_regression" in stats
        assert "feature_importance" in stats
        assert "minmax" in stats

    def test_json_serializable(self, runner, sample_csv, tmp_path):
        """Test that all output is JSON serializable."""
        result = runner.invoke(main, [
            'analyze', str(sample_csv),
            '--output-dir', str(tmp_path / 'output'),
            '--skip-llm',
            '-j'
        ])

        # Should not raise
        data = json.loads(result.output)

        # Re-serialize to ensure no issues
        json.dumps(data)
