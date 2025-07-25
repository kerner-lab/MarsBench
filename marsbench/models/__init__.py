"""
Model registry and import utilities for MarsBench.
"""

import logging

from omegaconf import DictConfig

from .classification import InceptionV3
from .classification import ResNet101
from .classification import SqueezeNet
from .classification import SwinTransformer
from .classification import ViT
from .detection import SSD
from .detection import FasterRCNN
from .detection import RetinaNet
from .segmentation import DPT
from .segmentation import DeepLab
from .segmentation import Mask2Former
from .segmentation import Segformer
from .segmentation import UNet

log = logging.getLogger(__name__)

# Map of model names to classes
MODEL_REGISTRY = {
    "classification": {
        "InceptionV3": InceptionV3,
        "ResNet101": ResNet101,
        "SqueezeNet": SqueezeNet,
        "SwinTransformer": SwinTransformer,
        "VisionTransformer": ViT,
    },
    "segmentation": {
        "UNet": UNet,
        "DeepLab": DeepLab,
        "Segformer": Segformer,
        "Mask2Former": Mask2Former,
        "DPT": DPT,
    },
    "detection": {
        "FasterRCNN": FasterRCNN,
        "RetinaNet": RetinaNet,
        "SSD": SSD,
    },
}


def import_model_class(cfg: DictConfig):
    """Import model class based on configuration.

    Args:
        cfg (DictConfig): Configuration containing model information

    Returns:
        The model class (not instantiated)

    Raises:
        ValueError: If model is not found
    """
    try:
        model_name = cfg.model.name
        if model_name not in MODEL_REGISTRY[cfg.task]:
            log.error(
                f"Model {model_name} not found in available models for "
                "{cfg.task}: {list(MODEL_REGISTRY[cfg.task].keys())}"
            )
            raise ValueError(f"Model {model_name} not found")

        # Return the model class (not instantiated)
        model_class = MODEL_REGISTRY[cfg.task][model_name]
        log.info(f"Successfully imported {model_name} class")
        return model_class

    except Exception as e:
        log.error(f"Failed to import model class: {str(e)}")
        raise
