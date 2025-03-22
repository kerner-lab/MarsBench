"""
Base class for all Mars surface image segmentation datasets.
"""

import logging
import os
from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class BaseSegmentationDataset(Dataset, ABC):
    """Abstract base class for custom datasets.

    Attributes:
        data_dir (str): Directory where data is stored.
        transform (callable, optional): A function/transform to apply to the images.
        mask_transform (callable, optional): A function/transform to apply to the masks.

    Methods:
        _load_data(): Abstract method to load image paths and labels. Must be overridden.
        __len__(): Returns the size of the dataset.
        __getitem__(index): Retrieves an image and its label at the specified index.

    Usage:
        class MyDataset(CustomDataset):
            def _load_data(self):
                # Implement data loading logic
                return image_paths, labels
    """

    def __init__(
        self,
        cfg: DictConfig,
        data_dir: str,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        mask_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        split: Literal["train", "val", "test"] = "train",
    ):
        self.cfg = cfg
        # Map image types to PIL modes
        IMAGE_MODES = {"rgb": "RGB", "grayscale": "L", "l": "L"}
        requested_mode = cfg.data.image_type.lower().strip()
        self.image_type = IMAGE_MODES.get(requested_mode)
        if self.image_type is None:
            logger.error(
                f"Invalid/unsupported image_type '{requested_mode}'. Valid options are: {list(IMAGE_MODES.keys())}. "
                "Defaulting to RGB."
            )
            self.image_type = "RGB"
        self.data_dir = data_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.split = split

        logger.info(f"Loading {self.__class__.__name__} from {data_dir} (split: {split})")
        self.image_paths, self.ground = self._load_data()
        logger.info(f"Loaded {len(self.image_paths)} image-mask pairs")

        # Validate image extensions
        for image_path in self.image_paths:
            if not image_path.endswith(tuple(cfg.data.valid_image_extensions)):
                logger.error(f"Invalid image format: {image_path}")
                raise ValueError(f"Invalid image format: {image_path}")

        logger.info(
            f"Dataset initialized with mode: {self.image_type}, "
            f"transforms: {'applied' if transform else 'none'}, "
            f"mask_transforms: {'applied' if mask_transform else 'none'}"
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

    def __getitem__(self, ind: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image as grayscale (single channel)
        image = Image.open(os.path.join(self.data_dir, self.image_paths[ind])).convert(self.image_type)

        # Load mask as grayscale (single channel)
        mask = Image.open(os.path.join(self.data_dir, self.ground[ind])).convert("L")

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Default mask transform if none provided
            mask = transforms.ToTensor()(mask)
            mask = mask.squeeze(0)  # Remove channel dimension, making it [H, W]

        return image, mask
