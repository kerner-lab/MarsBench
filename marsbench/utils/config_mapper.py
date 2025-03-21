"""
Utility functions for mapping configuration overrides to the correct config files.
"""

import logging
import os
from pathlib import Path
from typing import Optional
from typing import Union

from omegaconf import DictConfig
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class ConfigLoadError(Exception):
    """Exception raised for errors in loading configurations."""

    pass


def load_dynamic_configs(cfg: DictConfig, config_dir: Optional[Union[str, Path]] = None) -> DictConfig:
    """
    Load data and model configs dynamically based on task, data_name, and model_name.

    Args:
        cfg: Base configuration from Hydra
        config_dir: Path to config directory (will use Hydra's default if not provided)

    Returns:
        Updated configuration with data and model configs merged in

    Raises:
        ConfigLoadError: If a required configuration file is not found
    """
    # Clone the configuration to avoid modifying the original
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    # Get the config directory, defaulting to "configs" relative to working directory
    if config_dir is None:
        # Try to get from hydra context first
        if hasattr(cfg, "hydra") and hasattr(cfg.hydra, "runtime") and hasattr(cfg.hydra.runtime, "cwd"):
            config_dir = Path(cfg.hydra.runtime.cwd) / "configs"
        else:
            # Fallback to environment variable or default
            hydra_config_path = os.getenv("HYDRA_CONFIG_PATH", "configs")
            config_dir = Path(hydra_config_path)
    else:
        config_dir = Path(config_dir)

    logger.info(f"Using config directory: {config_dir}")

    # Get mappings based on variables
    mappings = {
        "data": f"{cfg.task}/{cfg.data_name}" if hasattr(cfg, "task") and hasattr(cfg, "data_name") else None,
        "model": f"{cfg.task}/{cfg.model_name}" if hasattr(cfg, "task") and hasattr(cfg, "model_name") else None,
    }

    # Track what we've loaded for logging
    loaded_configs = {}

    # Process each mapping
    for config_type, path_suffix in mappings.items():
        # Skip if mapping is None
        if path_suffix is None:
            logger.warning(f"Cannot determine {config_type} config path: missing required variables")
            continue

        # Always force reload when task or model_name/data_name changes
        should_reload = True
        should_preserve = False

        # Get the related name parameter (e.g., "model_name" for "model" type)
        name_param = f"{config_type}_name"  # e.g., "data_name" or "model_name"

        if hasattr(cfg, config_type) and getattr(cfg, config_type) is not None:
            config_value = getattr(cfg, config_type)

            # If the config has a custom attribute, it was likely manually set and should be preserved
            if hasattr(config_value, "custom_attribute") or (
                isinstance(config_value, dict) and "custom_attribute" in config_value
            ):
                logger.debug(f"{config_type.capitalize()} config has custom attributes, preserving original")
                should_preserve = True
                should_reload = False

            # Check if the config actually matches the current task and name
            elif hasattr(config_value, "name") and hasattr(cfg, name_param):
                # Log values for debugging
                expected_name = getattr(cfg, name_param).lower()
                current_name = config_value.name.lower() if hasattr(config_value, "name") else "unknown"
                logger.debug(
                    f"Checking if {config_type} needs reload: expected={expected_name}, current={current_name}"
                )

                # Skip reloading if name matches the expected parameter
                if current_name == expected_name:
                    logger.debug(
                        f"{config_type.capitalize()} config already matches "
                        f"{name_param}={expected_name}, skipping reload"
                    )
                    should_reload = False
                else:
                    # Only force reload if name or task differs
                    logger.debug(
                        f"{config_type.capitalize()} config doesn't match {name_param}, "
                        f"will reload: current={current_name}, expected={expected_name}"
                    )

        if not should_reload:
            logger.info(f"Skipping {config_type} config reload: existing config is appropriate")
            continue

        # Construct path and try to load
        config_path = config_dir / config_type / f"{path_suffix}.yaml"

        if not config_path.exists():
            msg = f"Cannot find {config_type} config at {config_path}"

            # In test environment, we might want to be more lenient
            if os.getenv("TEST_ENV") == "1":
                logger.warning(f"{msg} (in TEST_ENV, continuing without this config)")
                continue
            else:
                raise ConfigLoadError(msg)

        # Load the config from file
        logger.info(f"Loading {config_type} config from: {config_path}")
        type_cfg = OmegaConf.load(str(config_path))

        # Either set directly or merge based on what we determined earlier
        if (
            not hasattr(cfg, config_type)
            or getattr(cfg, config_type) is None
            or (isinstance(getattr(cfg, config_type), dict) and len(getattr(cfg, config_type)) == 0)
        ):
            # If config doesn't exist or is empty, set it directly
            setattr(cfg, config_type, type_cfg)
        elif should_preserve:
            # If we want to preserve existing config, don't do anything
            logger.debug(f"Preserving existing {config_type} config with custom attributes")
        else:
            # Otherwise merge the loaded config with any existing values
            existing_cfg = getattr(cfg, config_type)
            merged_cfg = OmegaConf.merge(existing_cfg, type_cfg)
            setattr(cfg, config_type, merged_cfg)

        loaded_configs[config_type] = str(config_path)

    if loaded_configs:
        logger.info(f"Dynamically loaded configs: {loaded_configs}")
    else:
        logger.warning("No dynamic configs were loaded")

    return cfg
