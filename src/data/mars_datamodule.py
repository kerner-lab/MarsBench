from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.data import get_dataset
from src.utils.transforms import get_transforms


class MarsDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(MarsDataModule, self).__init__()
        self.cfg = cfg
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self):
        # Download or process data if needed
        pass

    def setup(self, stage=None):
        transforms = get_transforms(self.cfg)
        if self.cfg.task == "classification":
            self.train_dataset, self.val_dataset, self.test_dataset = get_dataset(
                self.cfg, transforms[:2]
            )
        elif self.cfg.task == "segmentation":
            self.train_dataset, self.val_dataset, self.test_dataset = get_dataset(
                self.cfg, transforms[:2], mask_transforms=transforms[2:]
            )
        elif self.cfg.task == "detection":
            self.train_dataset, self.val_dataset, self.test_dataset = get_dataset(
                self.cfg, transforms[:2]
            )
        else:
            raise ValueError(f"Task not yet supported: {self.cfg.task}")

    @staticmethod
    def detection_collate_fn(batch):
        images, targets = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        return images, targets

    @staticmethod
    def detection_collate_fn_v2(batch):
        images, targets = tuple(zip(*batch))
        images = torch.stack(images, dim=0)

        boxes = [target["bbox"] for target in targets]
        labels = [target["cls"] for target in targets]
        img_sizes = torch.stack([target["img_size"] for target in targets])
        img_scales = torch.tensor([target["img_scale"] for target in targets])

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_sizes,
            "img_scale": img_scales,
        }
        return images, annotations

    def get_collate_fn(self):
        if self.cfg.task == "detection":
            if self.cfg.model.detection.name.lower() == "efficientdet":
                return MarsDataModule.detection_collate_fn_v2
            else:
                return MarsDataModule.detection_collate_fn
        else:
            return None

    def train_dataloader(self):
        assert self.train_dataset is not None, "train_dataset is not loaded."
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.training.num_workers,
            collate_fn=self.get_collate_fn(),
        )

    def val_dataloader(self):
        assert self.val_dataset is not None, "val_dataset is not loaded."
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            collate_fn=self.get_collate_fn(),
        )

    def test_dataloader(self):
        assert self.test_dataset is not None, "test_dataset is not loaded."
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            collate_fn=self.get_collate_fn(),
        )
