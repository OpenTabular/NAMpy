"""Shared path constants for the test suite."""

from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
PARITY_DIR = TESTS_DIR / "parity"
