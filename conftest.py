"""Pytest configuration: makes the project root importable (data, models, utils)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
