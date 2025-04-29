"""
Segmentation datasets for Mars surface image segmentation tasks.
"""
from .BaseSegmentationDataset import BaseSegmentationDataset
from .Boulder_Segmentation import Boulder_Segmentation
from .ConeQuest_Segmentation import ConeQuest_Segmentation
from .Crater_Binary_Segmentation import Crater_Binary_Segmentation
from .Crater_Multi_Segmentation import Crater_Multi_Segmentation
from .MarsSegMER import MarsSegMER
from .MarsSegMSL import MarsSegMSL
from .MMLS import MMLS
from .S5Mars import S5Mars

__all__ = [
    "BaseSegmentationDataset",
    "ConeQuest_Segmentation",
    "Boulder_Segmentation",
    "MarsSegMER",
    "MarsSegMSL",
    "MMLS",
    "S5Mars",
    "Crater_Binary_Segmentation",
    "Crater_Multi_Segmentation",
]
