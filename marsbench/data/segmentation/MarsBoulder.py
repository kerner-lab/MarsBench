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


class MarsBoulder(BaseSegmentationDataset):
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
        image_dir = os.path.join(self.data_dir, "data", self.split, "images")
        mask_dir = os.path.join(self.data_dir, "data", self.split, "masks")
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith("_image.tif")])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith("_segmask.tif")])

        # Match by base name (remove _image.tif / _segmask.tif)
        image_bases = {f.replace("_image.tif", ""): f for f in image_files}
        mask_bases = {f.replace("_segmask.tif", ""): f for f in mask_files}
        valid_keys = sorted(set(image_bases.keys()) & set(mask_bases.keys()))

        if len(valid_keys) == 0:
            logging.warning("No matching image/mask pairs found in MarsBoulder dataset.")

        image_paths = [os.path.join(image_dir, image_bases[k]) for k in valid_keys]
        mask_paths = [os.path.join(mask_dir, mask_bases[k]) for k in valid_keys]
        return image_paths, mask_paths
