"""
Mars Dust Devil dataset for Mars dust devil detection.
"""

from typing import Literal

from omegaconf import DictConfig

from .BaseDetectionDataset import BaseDetectionDataset


class Mars_Dust_Devil(BaseDetectionDataset):
    def __init__(
        self,
        cfg: DictConfig,
        data_dir: str,
        transform=None,
        bbox_format: Literal["coco", "yolo", "pascal_voc"] = "yolo",
        split: Literal["train", "val", "test"] = "train",
    ):
        super().__init__(cfg, data_dir, transform, bbox_format, split)
