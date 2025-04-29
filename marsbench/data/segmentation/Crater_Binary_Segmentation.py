"""
Crater dataset for Mars crater binary segmentation.
"""

import logging
import os
from typing import Callable
from typing import Literal
from typing import Optional

import torch
from omegaconf import DictConfig
from PIL import Image

from .BaseSegmentationDataset import BaseSegmentationDataset

logger = logging.getLogger(__name__)


class Crater_Binary_Segmentation(BaseSegmentationDataset):
    def __init__(
        self,
        cfg: DictConfig,
        data_dir: str,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        split: Literal["train", "val", "test"] = "train",
    ):
        super().__init__(cfg, data_dir, transform, split)

    def _load_data(self):
        """Load image and mask paths, logging any mismatches.

        Returns:
            tuple: Lists of image paths and corresponding mask paths that have matches
        """
        image_set = set(os.listdir(os.path.join(self.data_dir, "data", self.split, "images")))
        mask_set = set(os.listdir(os.path.join(self.data_dir, "data", self.split, "masks")))
        missing_masks = image_set - mask_set
        missing_images = mask_set - image_set

        if missing_masks:
            logging.warning(f"Missing {len(missing_masks)} masks for images")
        if missing_images:
            logging.warning(f"Missing {len(missing_images)} images for masks")

        valid_names = image_set.intersection(mask_set)
        valid_paths = sorted(list(valid_names))

        image_paths = [os.path.join(self.data_dir, "data", self.split, "images", p) for p in valid_paths]
        mask_paths = [os.path.join(self.data_dir, "data", self.split, "masks", p) for p in valid_paths]

        return image_paths, mask_paths
