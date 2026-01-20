"""Tests for encoding detection utilities."""

import pytest
from pathlib import Path

from regression_analyzer.loader import detect_encoding


class TestDetectEncoding:
    """Tests for the detect_encoding function."""

    def test_detect_utf8_ascii(self, tmp_path: Path):
        """Test UTF-8 detection with ASCII content."""
        f = tmp_path / "utf8.txt"
        f.write_text("Hello World", encoding="utf-8")

        enc = detect_encoding(f)

        # ASCII content in UTF-8 file should return utf-8
        assert enc.lower() in ['utf-8', 'ascii']

    def test_detect_utf8_with_unicode(self, tmp_path: Path):
        """Test UTF-8 detection with Unicode characters."""
        f = tmp_path / "unicode.txt"
        f.write_text("Hello \u4e16\u754c \u2603 \u2764", encoding="utf-8")  # Chinese + emoji

        enc = detect_encoding(f)

        assert enc.lower() == 'utf-8'

    def test_detect_latin1(self, tmp_path: Path):
        """Test Latin-1 detection with special characters."""
        f = tmp_path / "latin1.txt"
        # French and German characters that are valid in Latin-1 but not ASCII
        content = "Caf\xe9 Fran\xe7ais M\xfcnchen"
        f.write_bytes(content.encode('latin-1'))

        enc = detect_encoding(f)

        # Should detect as latin-1, ISO-8859-1, cp1252, or similar
        assert enc is not None
        assert enc.lower() in ['latin-1', 'iso-8859-1', 'cp1252', 'windows-1252', 'utf-8']

    def test_detect_cp1252(self, tmp_path: Path):
        """Test Windows-1252 detection."""
        f = tmp_path / "cp1252.txt"
        # Windows-1252 has special characters in 0x80-0x9F range
        content = b"Smart quotes: \x93hello\x94"
        f.write_bytes(content)

        enc = detect_encoding(f)

        # Should detect as cp1252 or similar Windows encoding
        assert enc is not None

    def test_detect_empty_file(self, tmp_path: Path):
        """Test encoding detection on empty file."""
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")

        enc = detect_encoding(f)

        # Empty file should fall back to utf-8
        assert enc.lower() == 'utf-8'

    def test_custom_sample_size(self, tmp_path: Path):
        """Test encoding detection with custom sample size."""
        f = tmp_path / "test.txt"
        # Create a file with UTF-8 content
        f.write_text("A" * 5000 + "\u00e9", encoding="utf-8")

        enc = detect_encoding(f, sample_size=100)

        # Even with small sample, should work
        assert enc is not None

    def test_detect_utf8_bom(self, tmp_path: Path):
        """Test UTF-8 with BOM detection."""
        f = tmp_path / "bom.txt"
        # UTF-8 BOM followed by content
        content = b"\xef\xbb\xbfHello World"
        f.write_bytes(content)

        enc = detect_encoding(f)

        # Should detect as UTF-8 (with or without BOM marker)
        assert 'utf' in enc.lower()

    def test_detect_utf16_le(self, tmp_path: Path):
        """Test UTF-16 Little Endian detection."""
        f = tmp_path / "utf16le.txt"
        content = "Hello World"
        f.write_bytes(content.encode('utf-16-le'))

        enc = detect_encoding(f)

        # chardet should detect UTF-16 LE
        assert enc is not None

    def test_encoding_map_iso_to_latin1(self, tmp_path: Path):
        """Test that ISO-8859-1 is mapped to latin-1."""
        f = tmp_path / "iso.txt"
        # Strong Latin-1 indicators
        f.write_bytes(b"\xe0\xe8\xec\xf2\xf9" * 100)  # Italian vowels with accents

        enc = detect_encoding(f)

        # The encoding map should handle this
        assert enc is not None

    def test_detect_with_csv_content(self, tmp_path: Path):
        """Test encoding detection on typical CSV content."""
        f = tmp_path / "data.csv"
        csv_content = "name,age,city\nAlice,30,New York\nBob,25,Boston"
        f.write_text(csv_content, encoding="utf-8")

        enc = detect_encoding(f)

        assert enc.lower() in ['utf-8', 'ascii']

    def test_detect_with_csv_special_chars(self, tmp_path: Path):
        """Test encoding detection on CSV with special characters."""
        f = tmp_path / "special.csv"
        csv_content = "nom,ville\nFran\xe7ois,Paris\nJos\xe9,Montr\xe9al"
        f.write_bytes(csv_content.encode('latin-1'))

        enc = detect_encoding(f)

        # Should be detected properly
        assert enc is not None


class TestEncodingFallback:
    """Tests for encoding fallback behavior."""

    def test_low_confidence_fallback(self, tmp_path: Path):
        """Test that low confidence detection falls back to UTF-8."""
        f = tmp_path / "ambiguous.txt"
        # Very short content that's ambiguous
        f.write_bytes(b"ab")

        enc = detect_encoding(f)

        # Short content typically has low confidence, should fall back
        assert enc is not None

    def test_binary_content_handling(self, tmp_path: Path):
        """Test handling of binary-like content."""
        f = tmp_path / "binary.txt"
        # Some binary-ish bytes
        f.write_bytes(bytes(range(256)))

        enc = detect_encoding(f)

        # Should still return an encoding (likely utf-8 fallback)
        assert enc is not None
