import multiprocessing
from typing import Optional

import pytorch_lightning as pl
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
        else:
            raise ValueError(f"Task not yet supported: {self.cfg.task}")

    def train_dataloader(self):
        """Get train dataloader."""
        assert self.train_dataset is not None, "train_dataset is not loaded."
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=min(2, multiprocessing.cpu_count()),  # Limit to suggested max
            pin_memory=True,
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        assert self.val_dataset is not None, "val_dataset is not loaded."
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=min(2, multiprocessing.cpu_count()),  # Limit to suggested max
            pin_memory=True,
        )

    def test_dataloader(self):
        """Get test dataloader."""
        assert self.test_dataset is not None, "test_dataset is not loaded."
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=min(2, multiprocessing.cpu_count()),  # Limit to suggested max
            pin_memory=True,
        )
