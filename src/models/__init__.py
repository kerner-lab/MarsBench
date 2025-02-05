from .classification import InceptionV3
from .classification import ResNet18
from .classification import ResNet50
from .classification import SqueezeNet
from .classification import SwinTransformer
from .classification import ViT
from .detection import FasterRCNN
from .segmentation import DeepLab
from .segmentation import UNet

__all__ = [
    "InceptionV3",
    "ResNet18",
    "ResNet50",
    "SqueezeNet",
    "SwinTransformer",
    "ViT",
    "UNet",
    "DeepLab",
    "FasterRCNN",
]
