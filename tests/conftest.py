"""
Conftest file for MarsBench tests.
"""

import os
import sys

import pytest

from marsbench.utils.config_mapper import load_dynamic_configs

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

    # Set TEST_ENV to true for test-specific configurations
    monkeypatch.setenv("TEST_ENV", "true")

    # Set TEST_ENV to true for test-specific configurations
    monkeypatch.setenv("TEST_ENV", "true")

    # Disable warnings in pytest
    import warnings

    # Filter out non-critical test warnings
    warnings.filterwarnings(
        "ignore",
        message="The `srun` command is available on your system but is not used",
    )

    # Filter out ResourceWarnings which are common with DataLoader workers
    warnings.filterwarnings("ignore", category=ResourceWarning)

    # Filter out other common test warnings
    warnings.filterwarnings("ignore", message=".*cuda initialization.*")
    warnings.filterwarnings("ignore", message=".*overflow encountered in exp.*")


# Load task-specific configs for testing
@pytest.fixture
def config():
    """Load default config for testing."""
    from hydra import compose
    from hydra import initialize

    with initialize(version_base=None, config_path="../marsbench/configs"):
        # Load the base config with minimal settings
        cfg = compose(config_name="config", overrides=["training=test"])
        cfg = load_dynamic_configs(cfg)
    return cfg


@pytest.fixture
def classification_config():
    """Load default classification config for testing."""
    from hydra import compose
    from hydra import initialize

    with initialize(version_base=None, config_path="../marsbench/configs"):
        # Load the base config and override with classification settings
        cfg = compose(
            config_name="config",
            overrides=["task=classification", "data_name=domars16k", "model_name=resnet18", "training=test"],
        )
        cfg = load_dynamic_configs(cfg)
    return cfg


@pytest.fixture
def segmentation_config():
    """Load default segmentation config for testing."""
    from hydra import compose
    from hydra import initialize

    with initialize(version_base=None, config_path="../marsbench/configs"):
        # Load the base config and override with segmentation settings
        cfg = compose(
            config_name="config",
            overrides=["task=segmentation", "data_name=cone_quest", "model_name=unet", "training=test"],
        )
        cfg = load_dynamic_configs(cfg)
    return cfg


# Decorator to skip tests that require local data in CI environment
def skip_if_ci(func):
    """Decorator to skip tests that require local data when running in CI environment."""
    return pytest.mark.skipif(
        os.environ.get("CI_MODE", "false").lower() == "true",
        reason="Test requires local data, skipping in CI environment",
    )(func)
