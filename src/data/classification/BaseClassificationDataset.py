from typing import Any, Callable, List, Optional, Tuple
import torch
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from omegaconf import DictConfig

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
    def __init__(self, cfg: DictConfig, data_dir: str, transform: Optional[Callable[[Image.Image], torch.Tensor]] = None):
        self.cfg = cfg
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths, self.labels = self._load_data()
        for image_path in self.image_paths:
            if not image_path.endswith(tuple(cfg.data.valid_image_extensions)):
                raise ValueError(f"Invalid image format: {image_path}")

    @abstractmethod
    def _load_data(self) -> Tuple[List[str], List[int]]:
        pass

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, ind) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[ind]).convert("RGB")
        label = self.labels[ind]

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label
