"""
Classification models for Mars surface image classification tasks.
"""
from .InceptionV3 import InceptionV3
from .ResNet18 import ResNet18
from .ResNet50 import ResNet50
from .SqueezeNet import SqueezeNet
from .SwinTransformer import SwinTransformer
from .ViT import ViT

__all__ = [
    "ResNet50",
    "ResNet18",
    "InceptionV3",
    "SqueezeNet",
    "SwinTransformer",
    "ViT",
]
