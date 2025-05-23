"""
Base class for all Mars surface image classification datasets.
"""

import ast
import logging
from abc import ABC
from abc import abstractmethod
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

from marsbench.utils.load_mapping import load_mapping

logger = logging.getLogger(__name__)


class BaseClassificationDataset(Dataset, ABC):
    """Abstract base class for custom datasets.

    Attributes:
        data_dir (str): Directory where data is stored.
        transform (callable, optional): A function/transform to apply to the images.

    Methods:
        _load_data(): Abstract method to load image paths and gts. Must be overridden.
        __len__(): Returns the size of the dataset.
        __getitem__(index): Retrieves an image and its gt at the specified index.

    Usage:
        class MyDataset(CustomDataset):
            def _load_data(self):
                # Implement data loading logic
                return image_paths, gts
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
        self.image_paths, self.gts = self._load_data()
        logger.info(f"Loaded {len(self.image_paths)} images")

        self.cfg.mapping = load_mapping(self.data_dir, cfg.data.num_classes)

        # Validate image extensions
        for image_path in self.image_paths:
            if not image_path.endswith(tuple(cfg.data.valid_image_extensions)):
                logger.error(f"Invalid image format: {image_path}")
                raise ValueError(f"Invalid image format: {image_path}")

        logger.info(
            f"Dataset initialized with mode: {self.image_type}, " f"transforms: {'applied' if transform else 'none'}"
        )

    @abstractmethod
    def _load_data(self) -> Tuple[List[str], List[int] | List[str]]:
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
        return len(self.gts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int | torch.Tensor]:
        image = np.array(Image.open(self.image_paths[idx]).convert(self.image_type))
        if self.cfg.data.subtask == "multilabel":
            label = torch.zeros(self.cfg.data.num_classes, dtype=torch.float32)
            label[ast.literal_eval(str(self.gts[idx]))] = 1
        elif self.cfg.data.subtask == "binary":
            label = torch.tensor(self.gts[idx], dtype=torch.float32)
        else:
            label = torch.tensor(self.gts[idx], dtype=torch.long)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image)

        return image, label
