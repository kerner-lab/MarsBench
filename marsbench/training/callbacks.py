"""
Utilities for setting up and configuring callbacks.
"""

import logging

from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

log = logging.getLogger(__name__)


def setup_callbacks(cfg: DictConfig) -> list:
    """Set up callbacks based on configuration.

    Args:
        cfg (DictConfig): Configuration containing callback parameters

    Returns:
        list: List of configured callbacks
    """
    callbacks: list = []

    # Early stopping callback
    if cfg.training.get("early_stopping_patience") and cfg.callbacks.get("early_stopping", {}).get("enabled", True):
        # Create a simple config dict without _target_ and enabled
        early_stopping_config = OmegaConf.to_container(cfg.callbacks.early_stopping, resolve=True)
        early_stopping_config.pop("_target_", None)
        early_stopping_config.pop("enabled", None)

        # Override patience from training config
        early_stopping_config["patience"] = cfg.training.early_stopping_patience

        callbacks.append(EarlyStopping(**early_stopping_config))

    # Model checkpoint callbacks
    if cfg.training.trainer.get("enable_checkpointing", True):
        # Best checkpoint
        if cfg.callbacks.get("best_checkpoint", {}).get("enabled", True):
            best_checkpoint_config = OmegaConf.to_container(cfg.callbacks.best_checkpoint, resolve=True)
            best_checkpoint_config.pop("_target_", None)
            best_checkpoint_config.pop("enabled", None)

            callbacks.append(ModelCheckpoint(**best_checkpoint_config))

        # Last checkpoint
        if cfg.callbacks.get("last_checkpoint", {}).get("enabled", True):
            last_checkpoint_config = OmegaConf.to_container(cfg.callbacks.last_checkpoint, resolve=True)
            last_checkpoint_config.pop("_target_", None)
            last_checkpoint_config.pop("enabled", None)

            callbacks.append(ModelCheckpoint(**last_checkpoint_config))

    return callbacks
