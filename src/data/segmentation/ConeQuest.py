import logging
import os
from pathlib import Path
from typing import Callable
from typing import Literal
from typing import Optional

import torch
from omegaconf import DictConfig
from PIL import Image

from .BaseSegmentationDataset import BaseSegmentationDataset


class ConeQuest(BaseSegmentationDataset):
    def __init__(
        self,
        cfg: DictConfig,
        data_dir: str,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        mask_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        split: Literal["train", "val", "test"] = "train",
    ):
        split = split
        data_dir = Path(data_dir) / split
        super().__init__(cfg, data_dir, transform, mask_transform, split)

    def _load_data(self):
        """Load image and mask paths, logging any mismatches.

        Returns:
            tuple: Lists of image paths and corresponding mask paths that have matches
        """
        image_paths = sorted(os.listdir(Path(self.data_dir) / "images"))
        mask_paths = sorted(os.listdir(Path(self.data_dir) / "masks"))

        # Check for mismatches
        image_set = set(image_paths)
        mask_set = set(mask_paths)
        missing_masks = image_set - mask_set
        missing_images = mask_set - image_set

        if missing_masks:
            logging.warning(f"Missing masks for images: {missing_masks}")
        if missing_images:
            logging.warning(f"Missing images for masks: {missing_images}")

        # Keep only matched pairs
        valid_names = image_set.intersection(mask_set)
        valid_paths = sorted(list(valid_names))

        # Convert to full paths
        image_paths = [os.path.join(self.data_dir, "images", p) for p in valid_paths]
        mask_paths = [os.path.join(self.data_dir, "masks", p) for p in valid_paths]

        return image_paths, mask_paths
