"""Comprehensive tests for the DataLoader class."""

import pytest
import polars as pl
from pathlib import Path

from regression_analyzer.loader import DataLoader


class TestDataLoaderInit:
    """Tests for DataLoader initialization and validation."""

    def test_init_with_valid_csv(self, tmp_path: Path):
        """Test initialization with a valid CSV file path."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3")

        loader = DataLoader(csv_file)

        assert loader.file_path == csv_file

    def test_init_with_string_path(self, tmp_path: Path):
        """Test initialization with a string path instead of Path object."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3")

        loader = DataLoader(str(csv_file))

        assert loader.file_path == csv_file

    def test_file_not_found(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            DataLoader("/nonexistent/path/file.csv")

    def test_unsupported_format_txt(self, tmp_path: Path):
        """Test ValueError for .txt file type."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("some data")

        with pytest.raises(ValueError, match="Unsupported file type"):
            DataLoader(txt_file)

    def test_unsupported_format_json(self, tmp_path: Path):
        """Test ValueError for .json file type."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"a": 1}')

        with pytest.raises(ValueError, match="Unsupported file type"):
            DataLoader(json_file)

    def test_supported_extensions(self):
        """Test that SUPPORTED_EXTENSIONS contains expected values."""
        assert ".csv" in DataLoader.SUPPORTED_EXTENSIONS
        assert ".xlsx" in DataLoader.SUPPORTED_EXTENSIONS
        assert ".xls" in DataLoader.SUPPORTED_EXTENSIONS


class TestLoadCSV:
    """Tests for loading CSV files."""

    def test_load_csv_utf8(self, tmp_path: Path):
        """Test loading a standard UTF-8 CSV file."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6")

        loader = DataLoader(csv_file)
        df = loader.load()

        assert df.shape == (2, 3)
        assert df.columns == ["a", "b", "c"]
        assert df["a"].to_list() == [1, 4]

    def test_load_csv_latin1(self, tmp_path: Path):
        """Test loading Latin-1 encoded CSV with special characters."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_bytes("name,city\nJosé,São Paulo\nMüller,München".encode('latin-1'))

        loader = DataLoader(csv_file)
        df = loader.load()

        assert df.shape == (2, 2)
        assert "José" in df["name"].to_list()
        assert "São Paulo" in df["city"].to_list()

    def test_load_csv_with_null_values(self, tmp_path: Path):
        """Test that standard null values are properly detected."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,,3\n4,NA,6\n7,NULL,9")

        loader = DataLoader(csv_file)
        df = loader.load()

        # Check that null values are recognized
        null_count = df["b"].null_count()
        assert null_count >= 2  # At least empty and NA should be null

    def test_load_csv_with_mixed_types(self, tmp_path: Path):
        """Test loading CSV with mixed data types."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,name,value,active\n1,Alice,10.5,true\n2,Bob,20.3,false")

        loader = DataLoader(csv_file)
        df = loader.load()

        assert df.shape == (2, 4)
        assert "id" in df.columns
        assert "name" in df.columns

    def test_load_csv_empty_file(self, tmp_path: Path):
        """Test loading CSV with only headers."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n")

        loader = DataLoader(csv_file)
        df = loader.load()

        assert df.columns == ["a", "b", "c"]
        assert df.height == 0

    def test_load_csv_single_column(self, tmp_path: Path):
        """Test loading CSV with a single column."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("values\n1\n2\n3\n4\n5")

        loader = DataLoader(csv_file)
        df = loader.load()

        assert df.shape == (5, 1)
        assert df.columns == ["values"]

    def test_load_csv_large_values(self, tmp_path: Path):
        """Test loading CSV with large numeric values."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,big_number\n1,9999999999999\n2,1234567890123")

        loader = DataLoader(csv_file)
        df = loader.load()

        assert df["big_number"].to_list() == [9999999999999, 1234567890123]


class TestLoadWithTranspose:
    """Tests for loading data with transpose option."""

    def test_load_with_transpose_true(self, tmp_path: Path):
        """Test loading CSV with transpose=True."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("metric,Q1,Q2,Q3\nrevenue,100,110,120\ncost,60,65,70")

        loader = DataLoader(csv_file)
        df = loader.load(transpose=True)

        # After transpose, Q1/Q2/Q3 become rows, metrics become columns
        assert "revenue" in df.columns
        assert "cost" in df.columns
        assert df.height == 3  # Q1, Q2, Q3

    def test_load_with_transpose_false(self, tmp_path: Path):
        """Test loading CSV with transpose=False (default)."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6")

        loader = DataLoader(csv_file)
        df = loader.load(transpose=False)

        assert df.shape == (2, 3)
        assert df.columns == ["a", "b", "c"]


class TestGetMetadata:
    """Tests for metadata extraction."""

    def test_get_metadata_basic(self, tmp_path: Path):
        """Test basic metadata extraction."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x,y,z\n1,2,3\n4,5,6\n7,8,9")

        loader = DataLoader(csv_file)
        df = loader.load()
        metadata = loader.get_metadata(df)

        assert metadata["rows"] == 3
        assert metadata["columns"] == 3
        assert metadata["column_names"] == ["x", "y", "z"]

    def test_get_metadata_with_nulls(self, tmp_path: Path):
        """Test metadata extraction with null values."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x,y\n1,2\n3,\n5,6")

        loader = DataLoader(csv_file)
        df = loader.load()
        metadata = loader.get_metadata(df)

        assert metadata["rows"] == 3
        assert metadata["columns"] == 2
        assert "x" in metadata["column_names"]
        assert "y" in metadata["column_names"]
        assert "null_counts" in metadata
        assert metadata["null_counts"]["y"] >= 1

    def test_get_metadata_dtypes(self, tmp_path: Path):
        """Test that dtypes are included in metadata."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,name,value\n1,Alice,10.5\n2,Bob,20.3")

        loader = DataLoader(csv_file)
        df = loader.load()
        metadata = loader.get_metadata(df)

        assert "dtypes" in metadata
        assert "id" in metadata["dtypes"]
        assert "name" in metadata["dtypes"]
        assert "value" in metadata["dtypes"]

    def test_get_metadata_sample_values(self, tmp_path: Path):
        """Test that sample values are included in metadata."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x\n1\n2\n3\n4\n5")

        loader = DataLoader(csv_file)
        df = loader.load()
        metadata = loader.get_metadata(df)

        assert "sample_values" in metadata
        assert "x" in metadata["sample_values"]
        # Should have first 3 values
        assert len(metadata["sample_values"]["x"]) == 3
        assert metadata["sample_values"]["x"] == [1, 2, 3]


class TestCaseSensitivity:
    """Tests for file extension case sensitivity."""

    def test_uppercase_csv_extension(self, tmp_path: Path):
        """Test that .CSV files are accepted."""
        csv_file = tmp_path / "test.CSV"
        csv_file.write_text("a,b\n1,2")

        loader = DataLoader(csv_file)
        df = loader.load()

        assert df.shape == (1, 2)

    def test_mixed_case_extension(self, tmp_path: Path):
        """Test that .Csv files are accepted."""
        csv_file = tmp_path / "test.Csv"
        csv_file.write_text("a,b\n1,2")

        loader = DataLoader(csv_file)
        df = loader.load()

        assert df.shape == (1, 2)
