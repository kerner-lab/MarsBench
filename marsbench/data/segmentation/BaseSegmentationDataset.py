"""
Base class for all Mars surface image segmentation datasets.
"""

import json
import logging
import os
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Callable
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseSegmentationDataset(Dataset, ABC):
    """Abstract base class for custom datasets.

    Attributes:
        data_dir (str): Directory where data is stored.
        transform (callable, optional): A function/transform to apply to the images.

    Methods:
        _load_data(): Abstract method to load image paths and labels. Must be overridden.
        __len__(): Returns the size of the dataset.
        __getitem__(index): Retrieves an image and its label at the specified index.
    """

    def __init__(
        self,
        cfg: DictConfig,
        data_dir: str,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        split: Literal["train", "val", "test"] = "train",
    ):
        self.cfg = cfg
        IMAGE_MODES = {"rgb": "RGB", "grayscale": "L", "l": "L"}
        requested_mode = cfg.data.image_type.lower().strip()
        self.image_type = IMAGE_MODES.get(requested_mode)
        if self.image_type is None:
            logger.error(
                f"Invalid/unsupported image_type '{requested_mode}'. Valid options are: {list(IMAGE_MODES.keys())}. "
                "Defaulting to RGB."
            )
            self.image_type = "RGB"
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split

        logger.info(f"Loading {self.__class__.__name__} from {data_dir} (split: {split})")
        self.image_paths, self.ground = self._load_data()
        logger.info(f"Loaded {len(self.image_paths)} image-mask pairs")

        if len(self.image_paths) == 0 or len(self.ground) == 0:
            logging.error("No matching image and mask pairs found")
            raise ValueError("No matching image and mask pairs found")

        if np.array(self.ground[0]).ndim == 4:
            logger.info("One-hot encoded masks detected. Converting to class indices.")
            logger.warning("Expected shape of ground truth: [N, C, H, W]")

        for image_path in self.image_paths:
            if not image_path.endswith(tuple(cfg.data.valid_image_extensions)):
                logger.error(f"Invalid image format: {image_path}")
                raise ValueError(f"Invalid image format: {image_path}")

        if os.path.exists(self.data_dir / "mapping.json"):
            with open(self.data_dir / "mapping.json", "r") as f:
                self.cfg.mapping = {
                    int(k): v.strip().lower().replace(" ", "_").replace("-", "_") for k, v in json.load(f).items()
                }

            logger.info(f"Loaded mapping from {self.data_dir / 'mapping.json'}")
            if len(self.cfg.mapping) != self.cfg.data.num_classes:
                logger.warning(
                    f"Number of classes in mapping ({len(self.cfg.mapping)}) does not "
                    f"match num_classes ({self.cfg.data.num_classes})"
                )
                self.cfg.mapping = None
        else:
            self.cfg.mapping = None

        logger.info(
            f"Dataset initialized with mode: {self.image_type}, " f"transforms: {'applied' if transform else 'none'}, "
        )

    @abstractmethod
    def _load_data(self) -> Tuple[List[str], List[int]]:
        pass

    def determine_data_splits(
        self,
        total_size: int,
        generator: torch.Generator,
        split: Literal["train", "val", "test"],
    ):
        train_size = int(self.cfg.data.split.train * total_size)
        val_size = int(self.cfg.data.split.val * total_size)
        indices = torch.randperm(total_size, generator=generator).tolist()
        if split == "train":
            return indices[:train_size]
        elif split == "val":
            return indices[train_size : train_size + val_size]
        else:
            return indices[train_size + val_size :]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = np.array(Image.open(self.image_paths[idx]).convert(self.image_type))
        mask = np.array(Image.open(self.ground[idx]).convert("L"))

        if len(mask.shape) == 4:  # One hot encoded mask [N, C, H, W] to class mask [N, H, W]
            mask = np.argmax(mask, axis=2)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)

        mask = mask.to(torch.int64)
        return image, mask
