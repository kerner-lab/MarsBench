"""
Utilities for setting up and configuring loggers.
"""

import logging

from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

log = logging.getLogger(__name__)


def setup_logger(name, config):
    """Helper function to set up a logger based on its configuration.

    Args:
        name (str): Logger name
        config (DictConfig): Logger configuration

    Returns:
        Logger or None: Configured logger instance or None if disabled
    """
    if not config.get("enabled", False):
        return None

    # Remove 'enabled' from kwargs
    kwargs = OmegaConf.to_container(config, resolve=True)
    kwargs.pop("enabled", None)

    # Create appropriate logger
    if name == "wandb":
        return WandbLogger(**kwargs)
    elif name == "mlflow":
        return MLFlowLogger(**kwargs)
    elif name == "tensorboard":
        return TensorBoardLogger(**kwargs)
    elif name == "csv":
        return CSVLogger(**kwargs)
    else:
        log.warning(f"Unknown logger type: {name}")
        return None


def setup_loggers(cfg: DictConfig):
    """Set up all loggers based on configuration.

    Args:
        cfg (DictConfig): Configuration containing logger settings

    Returns:
        list: List of configured logger instances
    """
    loggers = []

    # Set up each logger type
    for logger_name in ["wandb", "mlflow", "tensorboard", "csv"]:
        if logger_name in cfg.logger:
            logger = setup_logger(logger_name, cfg.logger[logger_name])
            if logger:
                loggers.append(logger)

    return loggers
