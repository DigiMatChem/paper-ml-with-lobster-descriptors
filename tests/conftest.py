"""
Fixtures for tests
"""

from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def data_dir():
    """
    Absolute path of data directory.
    """
    return Path(__file__).resolve().parents[1] / "data"
