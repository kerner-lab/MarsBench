import logging
import os

import hydra
from omegaconf import DictConfig

from .classification import InceptionV3
from .classification import ResNet18
from .classification import ResNet50
from .classification import SqueezeNet
from .classification import SwinTransformer
from .classification import ViT
from .segmentation import DeepLab
from .segmentation import UNet

log = logging.getLogger(__name__)

# Map of model names to classes
MODEL_REGISTRY = {
    "InceptionV3": InceptionV3,
    "ResNet18": ResNet18,
    "ResNet50": ResNet50,
    "SqueezeNet": SqueezeNet,
    "SwinTransformer": SwinTransformer,
    "ViT": ViT,
    "UNet": UNet,
    "DeepLab": DeepLab,
}


def import_model_class(cfg: DictConfig):
    """Import model class based on configuration.

    Args:
        cfg (DictConfig): Configuration containing model information

    Returns:
        model: Instantiated model

    Raises:
        ValueError: If model is not found
    """
    try:
        model_name = cfg.model.name
        if model_name not in MODEL_REGISTRY:
            log.error(f"Model {model_name} not found in available models: {list(MODEL_REGISTRY.keys())}")
            raise ValueError(f"Model {model_name} not found")

        # Initialize model
        model_class = MODEL_REGISTRY[model_name]
        model = model_class(cfg)
        log.info(f"Successfully initialized {model_name}")
        return model

    except Exception as e:
        log.error(f"Failed to import model: {str(e)}")
        raise


if __name__ == "__main__":
    config_dir = os.path.abspath("../configs")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "task=classification",
                "model_name=vit",
                "data_name=domars16k",
                "seed=0",
                "training.max_epochs=5",
                "training.batch_size=16",
            ],
        )
    import_model_class(cfg)
