"""
Training, testing, and prediction execution utilities.
"""
import logging

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from marsbench.data.mars_datamodule import MarsDataModule
from marsbench.training.results import save_benchmark_results
from marsbench.training.results import save_predictions

log = logging.getLogger(__name__)


def run_training(trainer: Trainer, model: pl.LightningModule, data_module: MarsDataModule, cfg: DictConfig):
    """Run model training.

    Args:
        trainer: PyTorch Lightning Trainer
        model: Model to train
        data_module: Data module for training
        cfg: Configuration
    """
    log.info("Starting training")
    trainer.fit(model, data_module)

    # If specified, run testing after training
    if cfg.get("test_after_training", False):
        log.info("Training completed, running testing as requested")
        run_testing(trainer, model, data_module, cfg)


def run_testing(trainer: Trainer, model: pl.LightningModule, data_module: MarsDataModule, cfg: DictConfig):
    """Run model testing/benchmarking.

    Args:
        trainer: PyTorch Lightning Trainer
        model: Model to test
        data_module: Data module for testing
        cfg: Configuration
    """
    log.info("Starting testing")
    results = trainer.test(model, data_module)
    save_benchmark_results(cfg, results)


def run_prediction(trainer: Trainer, model: pl.LightningModule, data_module: MarsDataModule, cfg: DictConfig):
    """Run model prediction.

    Args:
        trainer: PyTorch Lightning Trainer
        model: Model to use for prediction
        data_module: Data module for prediction
        cfg: Configuration
    """
    log.info("Starting prediction")
    predictions = trainer.predict(model, data_module)
    save_predictions(cfg, predictions)
