"""
Base class for all Mars surface image detection datasets.
"""

import logging
from abc import ABC
from typing import Callable
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import abstractmethod

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDetectionDataset(Dataset, ABC):
    def __init__(
        self,
        cfg: DictConfig,
        data_dir: str,
        transform: Optional[Callable[[Image.Image, dict], dict]] = None,
        bbox_format: Literal["coco", "yolo", "pascal_voc"] = None,
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
        self.data_dir = data_dir
        self.transform = transform
        self.bbox_format = bbox_format
        self.split = split

        logger.info(f"Loading {self.__class__.__name__} from {data_dir} (split: {split})")
        (
            self.image_paths,
            self.annotations,
            self.labels,
            _,  # image_ids
        ) = self._load_data()
        logger.info(f"Loaded {len(self.image_paths)} images with annotations")

        # Validate image extensions
        for image_path in self.image_paths:
            if not image_path.endswith(tuple(cfg.data.valid_image_extensions)):
                logger.error(f"Invalid image format: {image_path}")
                raise ValueError(f"Invalid image format: {image_path}")

        logger.info(
            f"Dataset initialized with mode: {self.image_type}, " f"transforms: {'applied' if transform else 'none'}, "
        )

    @abstractmethod
    def _load_data(self) -> Tuple[List[str], List[List[float]], List[List[int]], List[str]]:
        pass

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, dict]:
        image = Image.open(self.image_paths[idx]).convert(self.image_type)
        bboxes = self.annotations[idx]
        labels = self.labels[idx]

        img_width, img_height = image.size
        image = np.array(image)

        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=labels)

        image = transformed["image"]
        img_height, img_width = image.shape[-2:]
        bboxes = transformed["bboxes"]
        labels = transformed["class_labels"]
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.cfg.model.name.lower() == "efficientdet":
            bboxes = bboxes[:, [1, 0, 3, 2]]
            bbox_label = "bbox"
            class_label = "cls"
        else:
            bbox_label = "boxes"
            class_label = "labels"

        target = {
            bbox_label: bboxes,
            class_label: labels,
            "image_id": torch.tensor([idx]),
            "img_size": torch.tensor([img_height, img_width]),
            "img_scale": torch.tensor([1.0]),
        }

        return image, target
