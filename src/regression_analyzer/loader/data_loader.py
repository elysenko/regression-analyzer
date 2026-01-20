"""Data loading module for CSV and Excel files."""

from pathlib import Path

import polars as pl

from .encoding import detect_encoding


class DataLoader:
    """Load tabular data from CSV/Excel files."""

    SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

    def __init__(self, file_path: str | Path):
        """Initialize DataLoader with a file path.

        Args:
            file_path: Path to the data file (CSV or Excel).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is not supported.
        """
        self.file_path = Path(file_path)
        self._validate_file()

    def _validate_file(self) -> None:
        """Validate that the file exists and has a supported extension."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        suffix = self.file_path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")

    def load(self, transpose: bool = False) -> pl.DataFrame:
        """Load data from file into a Polars DataFrame.

        Args:
            transpose: If True, transpose the DataFrame after loading.

        Returns:
            Loaded DataFrame.
        """
        suffix = self.file_path.suffix.lower()
        if suffix == ".csv":
            df = self._load_csv()
        else:
            df = self._load_excel()

        if transpose:
            from .transpose import transpose_dataframe

            df = transpose_dataframe(df)

        return df

    def _load_csv(self) -> pl.DataFrame:
        """Load a CSV file with automatic encoding detection."""
        encoding = detect_encoding(self.file_path)
        try:
            return pl.read_csv(
                self.file_path,
                encoding=encoding,
                infer_schema_length=10000,
                null_values=["", "NA", "N/A", "null", "NULL", "None"],
            )
        except Exception:
            # Try fallback encodings
            for fallback_enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    return pl.read_csv(
                        self.file_path,
                        encoding=fallback_enc,
                        infer_schema_length=10000,
                    )
                except Exception:
                    continue
            raise ValueError("Failed to load CSV with any encoding")

    def _load_excel(self) -> pl.DataFrame:
        """Load an Excel file using pandas as intermediate."""
        import pandas as pd

        pdf = pd.read_excel(self.file_path, sheet_name=0, engine="openpyxl")
        return pl.from_pandas(pdf)

    def get_metadata(self, df: pl.DataFrame) -> dict:
        """Extract metadata from a DataFrame.

        Args:
            df: The DataFrame to analyze.

        Returns:
            Dictionary containing metadata about the DataFrame.
        """
        return {
            "rows": df.height,
            "columns": df.width,
            "column_names": df.columns,
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "null_counts": {col: df[col].null_count() for col in df.columns},
            "sample_values": {col: df[col].head(3).to_list() for col in df.columns},
        }
