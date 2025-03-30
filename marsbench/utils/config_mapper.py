"""
Utility functions for mapping configuration overrides to the correct config files.
"""

import logging
import os
from pathlib import Path
from typing import Optional
from typing import Union

from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class ConfigLoadError(Exception):
    """Exception raised when a configuration file cannot be loaded."""

    def __init__(self, message):
        logger.error(f"Configuration error: {message}")
        super().__init__(message)


def load_dynamic_configs(cfg: DictConfig, config_dir: Optional[Union[str, Path]] = None) -> DictConfig:
    """
    Dynamically load configuration files based on the task, data name, and model name.

    Args:
        cfg: The base configuration
        config_dir: Optional directory where configuration files are stored

    Returns:
        Updated configuration with data and model configs merged in

    Raises:
        ConfigLoadError: If a required configuration file is not found
    """
    # Clone the configuration to avoid modifying the original
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    # Get the config directory, defaulting to "configs" relative to working directory
    if config_dir is None:
        try:
            original_cwd = get_original_cwd()
            config_dir = Path(original_cwd) / "configs"
        except Exception:
            hydra_config_path = os.getenv("HYDRA_CONFIG_PATH", "configs")
            config_dir = Path(hydra_config_path)
    else:
        config_dir = Path(config_dir)

    if not config_dir.exists():
        raise ConfigLoadError(f"Config directory {config_dir} does not exist")

    logger.info(f"Using config directory: {config_dir}")

    # Define mappings for config types
    mappings = {
        "data": f"{cfg.task}/{cfg.data_name}" if hasattr(cfg, "task") and hasattr(cfg, "data_name") else None,
        "model": f"{cfg.task}/{cfg.model_name}" if hasattr(cfg, "task") and hasattr(cfg, "model_name") else None,
    }

    # Track loaded configs for logging
    loaded_configs = {}

    # Check if data parameters are being overridden and warn users
    if hasattr(cfg, "data") and isinstance(cfg.data, dict) and len(cfg.data) > 0:
        logger.warning(
            "We don't support overriding data parameters via command line. "
            "For better consistency, consider editing the data config files directly."
        )

    # Check if model parameters are being overridden and warn users
    if hasattr(cfg, "model") and isinstance(cfg.model, dict) and len(cfg.model) > 0:
        logger.warning(
            "We don't support overriding model parameters via command line. "
            "For better consistency, consider editing the model config files directly."
        )

    # Process each config type
    for config_type, path_suffix in mappings.items():
        # Skip if mapping is None
        if path_suffix is None:
            logger.warning(f"Cannot determine {config_type} config path: missing required variables")
            continue

        # Construct path and try to load
        config_path = config_dir / config_type / f"{path_suffix}.yaml"

        if not config_path.exists():
            msg = f"Cannot find {config_type} config at {config_path}"
            # In test environment, we might want to be more lenient
            if os.getenv("TEST_ENV") == "1":
                logger.warning(f"{msg}, but continuing in test environment")
                continue
            else:
                raise ConfigLoadError(msg)

        # Load the config from file
        logger.info(f"Loading {config_type} config from: {config_path}")
        type_cfg = OmegaConf.load(str(config_path))

        # Merge with existing config or set directly
        if not hasattr(cfg, config_type) or getattr(cfg, config_type) is None:
            setattr(cfg, config_type, type_cfg)
        else:
            # Merge with existing config
            existing_cfg = getattr(cfg, config_type)
            merged_cfg = OmegaConf.merge(existing_cfg, type_cfg)
            setattr(cfg, config_type, merged_cfg)

        loaded_configs[config_type] = str(config_path)

    if loaded_configs:
        logger.info(f"Dynamically loaded configs: {loaded_configs}")

    return cfg
