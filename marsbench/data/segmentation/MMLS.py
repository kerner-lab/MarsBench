"""
MarsLS-Net: Martian Landslides Binary Segmentation Network and Benchmark Dataset
"""


import logging
import os
from pathlib import Path
from typing import Callable
from typing import Literal
from typing import Optional
from typing import Tuple

import tifffile
import torch
from omegaconf import DictConfig
from PIL import Image

from .BaseSegmentationDataset import BaseSegmentationDataset


class MMLS(BaseSegmentationDataset):
    def __init__(
        self,
        cfg: DictConfig,
        data_dir: str,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        split: Literal["train", "val", "test"] = "train",
    ):
        data_dir = Path(data_dir) / split
        super().__init__(cfg, data_dir, transform, split)

    def _load_data(self):
        """Load image and mask paths, logging any mismatches.

        Returns:
            tuple: Lists of image paths and corresponding mask paths that have matches
        """
        image_set = set([x.replace("Image", "") for x in os.listdir(Path(self.data_dir) / "images")])
        mask_set = set([x.replace("Mask", "") for x in os.listdir(Path(self.data_dir) / "masks")])
        missing_masks = image_set - mask_set
        missing_images = mask_set - image_set

        if missing_masks:
            logging.warning(f"Missing {len(missing_masks)} masks for images: {sorted(missing_masks)[:5]}")
        if missing_images:
            logging.warning(f"Missing {len(missing_images)} images for masks: {sorted(missing_images)[:5]}")

        valid_names = image_set.intersection(mask_set)
        valid_paths = sorted(list(valid_names))

        image_paths = [os.path.join(self.data_dir, "images", "Image" + p) for p in valid_paths]
        mask_paths = [os.path.join(self.data_dir, "masks", "Mask" + p) for p in valid_paths]
        return image_paths, mask_paths

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Original Image Dimensions (128, 128, 7); we will only use the first 3 channels (BGR) -> (RGB)
        image = tifffile.imread(self.image_paths[idx])[:, :, [2, 1, 0]]
        mask = tifffile.imread(self.ground[idx])

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)

        mask = mask.to(torch.int64)
        return image, mask
