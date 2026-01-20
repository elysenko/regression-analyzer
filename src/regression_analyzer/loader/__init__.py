"""Data loading utilities for regression analyzer."""

from .data_loader import DataLoader
from .encoding import detect_encoding

__all__ = ["DataLoader", "detect_encoding"]
