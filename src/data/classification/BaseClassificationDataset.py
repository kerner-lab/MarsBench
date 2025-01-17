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


class BaseClassificationDataset(Dataset, ABC):
    """Abstract base class for custom datasets.

    Attributes:
        data_dir (str): Directory where data is stored.
        transform (callable, optional): A function/transform to apply to the images.

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
        logger.info(f"Loading {self.__class__.__name__} from {data_dir}")
        self.image_paths, self.labels = self._load_data()
        logger.info(
            f"Loaded {len(self.image_paths)} images with {len(set(self.labels))} unique classes"
        )

        # Validate image extensions
        for image_path in self.image_paths:
            if not image_path.endswith(tuple(cfg.data.valid_image_extensions)):
                logger.error(f"Invalid image format: {image_path}")
                raise ValueError(f"Invalid image format: {image_path}")

        logger.info(
            f"Dataset initialized with mode: {self.image_type}, "
            f"transforms: {'applied' if transform else 'none'}"
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
        return len(self.labels)

    def __getitem__(self, ind) -> Tuple[torch.Tensor, int]:
        image = Image.open(os.path.join(self.data_dir, self.image_paths[ind])).convert(
            self.image_type
        )
        label = self.labels[ind]

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label
