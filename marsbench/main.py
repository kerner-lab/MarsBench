"""
Main entry point for MarsBench training, testing, and prediction pipelines.
"""
import logging

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from marsbench.data.mars_datamodule import MarsDataModule
from marsbench.training import run_prediction
from marsbench.training import run_testing
from marsbench.training import run_training
from marsbench.training import setup_callbacks
from marsbench.training import setup_model
from marsbench.utils.config_mapper import load_dynamic_configs
from marsbench.utils.logger import setup_loggers
from marsbench.utils.seed import seed_everything

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training pipeline.

    Args:
        cfg (DictConfig): Hydra configuration
    """
    try:
        # Load dynamic configs
        cfg = load_dynamic_configs(cfg)

        # Set up seed for reproducibility
        if "seed" in cfg:
            log.info(f"Setting seed to {cfg.seed}")
            seed_everything(cfg.seed)

        # Set up model (new or pre-trained)
        model = setup_model(cfg)

        # Create data module
        data_module = MarsDataModule(cfg)

        # Setup callbacks and loggers
        callbacks = setup_callbacks(cfg)
        loggers = setup_loggers(cfg)

        # Initialize trainer
        trainer_config = {k: v for k, v in cfg.training.trainer.items() if k not in ["logger"]}
        trainer = Trainer(
            callbacks=callbacks,
            logger=loggers,
            default_root_dir=hydra.utils.get_original_cwd(),
            **trainer_config,
        )

        # Dispatch to appropriate mode handler
        if cfg.mode == "train":
            run_training(trainer, model, data_module, cfg)
        elif cfg.mode == "test":
            run_testing(trainer, model, data_module, cfg)
        elif cfg.mode == "predict":
            run_prediction(trainer, model, data_module, cfg)
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")

    except Exception as e:
        log.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
