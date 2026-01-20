---
name: data-loader
id: PRD-01
description: CSV/Excel data loading with encoding detection and transpose support
status: backlog
phase: mvp
priority: P0
complexity: medium
wave: 1

depends_on: []

creates:
  - src/regression_analyzer/loader/__init__.py
  - src/regression_analyzer/loader/data_loader.py
  - src/regression_analyzer/loader/encoding.py
  - src/regression_analyzer/loader/transpose.py
  - tests/loader/test_data_loader.py
  - tests/loader/test_encoding.py
  - tests/loader/test_transpose.py

modifies: []

database:
  creates: []
  modifies: []

test_command: pytest tests/loader/

blocks: [PRD-02, PRD-03, PRD-04, PRD-05]

created: 2026-01-20T19:06:34Z
updated: 2026-01-20T19:06:34Z
---

# PRD-01: Data Loader

## Overview

**Feature:** CSV/Excel data loading with encoding detection and transpose support
**Priority:** P0 (Foundation - all other components depend on this)
**Complexity:** Medium
**Dependencies:** None

---

## Problem Statement

Tabular data comes in various formats with inconsistent encoding, structure, and orientation. The system needs to:
1. Load CSV and Excel files reliably
2. Handle encoding issues (UTF-8, Latin-1, Windows-1252, etc.)
3. Detect and apply transpose when rows/columns are inverted
4. Provide clean Polars DataFrames for downstream analysis

---

## Goals

1. Load CSV files with automatic encoding detection using `chardet`
2. Load Excel files (.xlsx, .xls) using `openpyxl`
3. Provide transpose function that can invert rows/columns
4. Return data as Polars DataFrame (with pandas fallback)
5. Handle edge cases: empty files, malformed data, mixed types
6. Expose metadata about loaded data (shape, dtypes, null counts)

---

## Non-Goals

- Database connections (CSV/Excel only for MVP)
- Streaming large files (load entirely into memory)
- Multi-sheet Excel handling (first sheet only)
- Data cleaning/transformation (separate concern)

---

## Technical Design

### Architecture

```
loader/
├── __init__.py          # Public API exports
├── data_loader.py       # Main DataLoader class
├── encoding.py          # Encoding detection utilities
└── transpose.py         # Transpose detection and application
```

### Core Classes

```python
# src/regression_analyzer/loader/data_loader.py

from pathlib import Path
from typing import Optional, Tuple
import polars as pl

from .encoding import detect_encoding
from .transpose import transpose_dataframe

class DataLoader:
    """Load tabular data from CSV/Excel files."""

    def __init__(self, file_path: str | Path):
        """Initialize loader with file path.

        Args:
            file_path: Path to CSV or Excel file
        """
        self.file_path = Path(file_path)
        self._validate_file()

    def _validate_file(self) -> None:
        """Validate file exists and has supported extension."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        suffix = self.file_path.suffix.lower()
        if suffix not in {'.csv', '.xlsx', '.xls'}:
            raise ValueError(f"Unsupported file type: {suffix}")

    def load(self, transpose: bool = False) -> pl.DataFrame:
        """Load data from file.

        Args:
            transpose: If True, transpose rows and columns

        Returns:
            Polars DataFrame with loaded data
        """
        suffix = self.file_path.suffix.lower()

        if suffix == '.csv':
            df = self._load_csv()
        else:
            df = self._load_excel()

        if transpose:
            df = transpose_dataframe(df)

        return df

    def _load_csv(self) -> pl.DataFrame:
        """Load CSV with encoding detection."""
        encoding = detect_encoding(self.file_path)

        try:
            return pl.read_csv(
                self.file_path,
                encoding=encoding,
                infer_schema_length=10000,
                null_values=["", "NA", "N/A", "null", "NULL", "None"],
            )
        except Exception as e:
            # Fallback: try common encodings
            for fallback_enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    return pl.read_csv(
                        self.file_path,
                        encoding=fallback_enc,
                        infer_schema_length=10000,
                    )
                except:
                    continue
            raise ValueError(f"Failed to load CSV: {e}")

    def _load_excel(self) -> pl.DataFrame:
        """Load Excel file (first sheet)."""
        import pandas as pd

        # Use pandas for Excel, then convert to Polars
        pdf = pd.read_excel(
            self.file_path,
            sheet_name=0,
            engine='openpyxl'
        )

        return pl.from_pandas(pdf)

    def get_metadata(self, df: pl.DataFrame) -> dict:
        """Get metadata about loaded DataFrame.

        Args:
            df: Loaded DataFrame

        Returns:
            Dict with shape, dtypes, null counts, sample values
        """
        return {
            "rows": df.height,
            "columns": df.width,
            "column_names": df.columns,
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "null_counts": {col: df[col].null_count() for col in df.columns},
            "sample_values": {
                col: df[col].head(3).to_list()
                for col in df.columns
            },
        }
```

### Encoding Detection

```python
# src/regression_analyzer/loader/encoding.py

from pathlib import Path
import chardet

def detect_encoding(file_path: Path, sample_size: int = 10000) -> str:
    """Detect file encoding using chardet.

    Args:
        file_path: Path to file
        sample_size: Bytes to sample for detection

    Returns:
        Detected encoding string (e.g., 'utf-8', 'latin-1')
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(sample_size)

    result = chardet.detect(raw_data)
    encoding = result.get('encoding', 'utf-8')
    confidence = result.get('confidence', 0)

    # Default to UTF-8 if confidence is low
    if confidence < 0.7:
        encoding = 'utf-8'

    # Normalize encoding names
    encoding_map = {
        'ascii': 'utf-8',
        'ISO-8859-1': 'latin-1',
        'Windows-1252': 'cp1252',
    }

    return encoding_map.get(encoding, encoding)
```

### Transpose Utility

```python
# src/regression_analyzer/loader/transpose.py

import polars as pl

def transpose_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Transpose DataFrame, converting first column to headers.

    Args:
        df: Input DataFrame where first column contains future headers

    Returns:
        Transposed DataFrame with first column as new headers
    """
    # Get the first column values as new headers
    new_headers = df.select(df.columns[0]).to_series().to_list()
    new_headers = [str(h) for h in new_headers]  # Ensure strings

    # Get remaining columns as data rows
    data_cols = df.columns[1:]

    # Build transposed data
    transposed_data = {}

    # First column of transposed = original column names (minus first)
    transposed_data["index"] = data_cols

    # Each original row becomes a column
    for i, header in enumerate(new_headers):
        # Make header unique if duplicate
        unique_header = header
        counter = 1
        while unique_header in transposed_data:
            unique_header = f"{header}_{counter}"
            counter += 1

        transposed_data[unique_header] = [
            df[col][i] for col in data_cols
        ]

    return pl.DataFrame(transposed_data)


def should_transpose_heuristic(df: pl.DataFrame) -> bool:
    """Simple heuristic to guess if transpose is needed.

    This is a fallback - LLM analysis is preferred.

    Args:
        df: Input DataFrame

    Returns:
        True if transpose might be beneficial
    """
    # Heuristic: if more rows than columns by 10x, might need transpose
    if df.height > df.width * 10:
        return False  # Normal orientation

    # Heuristic: if first column looks like headers (short strings, unique)
    first_col = df.select(df.columns[0]).to_series()

    if first_col.dtype == pl.Utf8:
        unique_ratio = first_col.n_unique() / len(first_col)
        avg_len = first_col.str.lengths().mean()

        # If high uniqueness and short strings, likely headers
        if unique_ratio > 0.9 and avg_len < 30:
            return True

    return False
```

---

## Implementation Tasks

### Task 1: Core Data Loader
- [ ] Create `loader/` directory structure
- [ ] Implement `DataLoader` class with CSV support
- [ ] Add encoding detection using chardet
- [ ] Add fallback encoding handling

### Task 2: Excel Support
- [ ] Add Excel loading via pandas + openpyxl
- [ ] Handle first sheet extraction
- [ ] Convert to Polars DataFrame

### Task 3: Transpose Functionality
- [ ] Implement `transpose_dataframe()` function
- [ ] Add heuristic `should_transpose_heuristic()`
- [ ] Handle duplicate column names

### Task 4: Metadata Extraction
- [ ] Implement `get_metadata()` method
- [ ] Include shape, dtypes, null counts
- [ ] Add sample values for each column

### Task 5: Testing
- [ ] Unit tests for CSV loading (various encodings)
- [ ] Unit tests for Excel loading
- [ ] Unit tests for transpose function
- [ ] Integration tests with sample files

---

## Testing Strategy

### Unit Tests
```python
# tests/loader/test_data_loader.py

import pytest
import polars as pl
from regression_analyzer.loader import DataLoader

def test_load_csv_utf8(tmp_path):
    """Test loading UTF-8 CSV."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("a,b,c\n1,2,3\n4,5,6")

    loader = DataLoader(csv_file)
    df = loader.load()

    assert df.shape == (2, 3)
    assert df.columns == ["a", "b", "c"]

def test_load_csv_latin1(tmp_path):
    """Test loading Latin-1 encoded CSV."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_bytes("name,city\nJosé,São Paulo".encode('latin-1'))

    loader = DataLoader(csv_file)
    df = loader.load()

    assert "José" in df["name"].to_list()

def test_transpose():
    """Test transpose functionality."""
    df = pl.DataFrame({
        "metric": ["revenue", "cost", "profit"],
        "Q1": [100, 60, 40],
        "Q2": [120, 70, 50],
    })

    loader = DataLoader.__new__(DataLoader)
    from regression_analyzer.loader.transpose import transpose_dataframe

    transposed = transpose_dataframe(df)

    assert "revenue" in transposed.columns
    assert transposed.height == 2  # Q1, Q2 rows
```

---

## Acceptance Criteria

1. **CSV Loading**: Loads CSV files with UTF-8, Latin-1, and Windows-1252 encodings
2. **Excel Loading**: Loads .xlsx files and extracts first sheet
3. **Transpose**: Can transpose DataFrame with first column as headers
4. **Metadata**: Returns accurate shape, dtypes, and null counts
5. **Error Handling**: Clear errors for missing files, unsupported formats
6. **Performance**: Loads 100K row CSV in under 2 seconds

---

## Dependencies

- `polars>=0.20.0` - Primary DataFrame library
- `pandas>=2.0.0` - Excel loading bridge
- `openpyxl>=3.1.0` - Excel file parsing
- `chardet>=5.0.0` - Encoding detection

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Encoding detection fails | Fallback chain: detected → utf-8 → latin-1 → cp1252 |
| Large files cause OOM | Document memory requirements, defer streaming to future |
| Transpose loses data types | Cast all to string during transpose, re-infer after |
