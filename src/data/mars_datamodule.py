import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from src.data import get_dataset
from src.utils.transforms import get_transforms
from typing import Optional

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
        train_transform, val_transform = get_transforms(self.cfg)
        self.train_dataset, self.val_dataset, self.test_dataset = get_dataset(
            self.cfg, train_transform, val_transform
        )

    def train_dataloader(self):
        assert self.train_dataset is not None, "train_dataset is not loaded."
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.training.num_workers
        )

    def val_dataloader(self):
        assert self.val_dataset is not None, "val_dataset is not loaded."
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers
        )

    def test_dataloader(self):
        assert self.test_dataset is not None, "test_dataset is not loaded."
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers
        )
