"""
Segmentation datasets for Mars surface image segmentation tasks.
"""
from .BaseSegmentationDataset import BaseSegmentationDataset
from .ConeQuest import ConeQuest
from .MarsBoulder import MarsBoulder
from .MarsData import MarsData
from .MarsSegMER import MarsSegMER
from .MarsSegMSL import MarsSegMSL
from .MMLS import MMLS
from .S5Mars import S5Mars

__all__ = [
    "BaseSegmentationDataset",
    "ConeQuest",
    "MarsBoulder",
    "MarsData",
    "MarsSegMER",
    "MarsSegMSL",
    "MMLS",
    "S5Mars",
]
