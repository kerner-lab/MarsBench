"""
Segmentation datasets for Mars surface image segmentation tasks.
"""
from .BaseSegmentationDataset import BaseSegmentationDataset
from .ConeQuest import ConeQuest

__all__ = ["BaseSegmentationDataset", "ConeQuest"]
