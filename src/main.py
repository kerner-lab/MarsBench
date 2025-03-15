import logging

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from .data.mars_datamodule import MarsDataModule
from .models import import_model_class
from .utils.config_mapper import load_dynamic_configs
from .utils.seed import seed_everything

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training pipeline.

    Args:
        cfg (DictConfig): Hydra configuration
    """
    try:
        # Update the configuration with dynamically loaded configs based on task, data_name, and model_name
        # This handles loading the appropriate data and model configs automatically
        cfg = load_dynamic_configs(cfg)

        # Set seed if provided
        if cfg.get("seed"):
            seed_everything(cfg.seed)

        # Create components
        data_module = MarsDataModule(cfg)
        model = import_model_class(cfg)

        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor="val/loss", patience=5, mode="min"),
            ModelCheckpoint(
                monitor="val/loss",
                save_top_k=1,
                mode="min",
                filename="best-{epoch:02d}-{val_loss:.4f}",
            ),
            ModelCheckpoint(save_last=True, save_top_k=0, filename="last"),
        ]

        # Setup loggers
        loggers = []
        if cfg.logger.wandb.enabled:
            loggers.append(
                WandbLogger(
                    project=cfg.logger.wandb.project,
                    name=cfg.logger.wandb.name,
                    entity=cfg.logger.wandb.entity,
                    tags=cfg.logger.wandb.tags,
                    notes=cfg.logger.wandb.notes,
                    save_code=cfg.logger.wandb.save_code,
                    mode=cfg.logger.wandb.mode,
                )
            )

        if cfg.logger.mlflow.enabled:
            loggers.append(
                MLFlowLogger(
                    experiment_name=cfg.logger.mlflow.experiment_name,
                    tracking_uri=cfg.logger.mlflow.tracking_uri,
                    run_name=cfg.logger.mlflow.run_name,
                    tags=cfg.logger.mlflow.tags,
                    save_dir=cfg.logger.mlflow.save_dir,
                )
            )

        if cfg.logger.tensorboard.enabled:
            loggers.append(
                TensorBoardLogger(
                    save_dir=cfg.logger.tensorboard.save_dir,
                    name=cfg.logger.tensorboard.name,
                    version=cfg.logger.tensorboard.version,
                )
            )

        if cfg.logger.csv.enabled:
            loggers.append(
                CSVLogger(
                    save_dir=cfg.logger.csv.save_dir,
                    name=cfg.logger.csv.name,
                    version=cfg.logger.csv.version,
                )
            )

        # Initialize trainer
        trainer_config = {k: v for k, v in cfg.training.trainer.items() if k not in ["logger"]}
        trainer = Trainer(
            callbacks=callbacks,
            logger=loggers,
            default_root_dir=hydra.utils.get_original_cwd(),
            **trainer_config,
        )

        # Run based on mode
        if cfg.mode == "train":
            trainer.fit(model, data_module)
        elif cfg.mode == "test":
            trainer.test(model, data_module)
        elif cfg.mode == "predict":
            trainer.predict(model, data_module)
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")

    except Exception as e:
        log.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
