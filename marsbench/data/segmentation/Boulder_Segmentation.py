"""
MarsBoulder dataset for binary segmentation of boulders.
"""

import logging
import os
from typing import Callable
from typing import Literal
from typing import Optional

from omegaconf import DictConfig

from .BaseSegmentationDataset import BaseSegmentationDataset


class Boulder_Segmentation(BaseSegmentationDataset):
    def __init__(
        self,
        cfg: DictConfig,
        data_dir: str,
        transform: Optional[Callable] = None,
        split: Literal["train", "val", "test"] = "train",
    ):
        super().__init__(cfg, data_dir, transform, split)

    def _load_data(self):
        """
        Loads image and mask paths for the Mars Boulder dataset.
        Assumes images are in 'images/' and masks in 'masks/' with matching base names.
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
