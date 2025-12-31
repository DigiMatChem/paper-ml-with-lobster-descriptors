"""
Fixtures for tests
"""
import psutil
from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def data_dir():
    """
    Absolute path of data directory.
    """
    return Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def test_data_dir():
    """
    Absolute path of test_data directory.
    """
    module_dir = Path(__file__).parent.resolve()
    test_dir = module_dir.joinpath("test_data")
    return test_dir.resolve()

@pytest.fixture(scope="session")
def num_jobs():
    """
    Number of processors available for testing.
    """
    return psutil.cpu_count(logical=False) or 1