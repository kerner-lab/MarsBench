import os
import sys
from pathlib import Path

import pytest

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the project root to sys.path
sys.path.insert(0, project_root)


# Override Hydra config for tests
@pytest.fixture(autouse=True)
def setup_test_config(monkeypatch):
    """Set environment variables to use test configuration for all tests."""
    monkeypatch.setenv("HYDRA_CONFIG_PATH", "configs")
    monkeypatch.setenv("HYDRA_CONFIG_NAME", "config")

    # Force test training config
    monkeypatch.setenv("HYDRA_TRAINING", "test")

    # Disable warnings in pytest
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="The `srun` command is available on your system but is not used",
    )
