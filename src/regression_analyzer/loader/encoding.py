"""Encoding detection utilities for file loading."""

from pathlib import Path

import chardet


def detect_encoding(file_path: Path, sample_size: int = 10000) -> str:
    """Detect file encoding using chardet.

    Args:
        file_path: Path to the file to analyze.
        sample_size: Number of bytes to read for detection.

    Returns:
        Detected encoding string suitable for file reading.
    """
    with open(file_path, "rb") as f:
        raw_data = f.read(sample_size)

    result = chardet.detect(raw_data)
    encoding = result.get("encoding", "utf-8")
    confidence = result.get("confidence", 0)

    # Fall back to utf-8 if confidence is low
    if confidence < 0.7:
        encoding = "utf-8"

    # Map common encodings to their Python-friendly names
    encoding_map = {
        "ascii": "utf-8",
        "ISO-8859-1": "latin-1",
        "Windows-1252": "cp1252",
    }

    return encoding_map.get(encoding, encoding)
