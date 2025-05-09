"""
PyTorch Lightning data module for Mars datasets.
"""

import os
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import Mask2FormerImageProcessor

from marsbench.data import get_dataset
from marsbench.utils.transforms import get_transforms


class MarsDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(MarsDataModule, self).__init__()
        self.cfg = cfg
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.image_processor = None

        sys_worker = os.environ.get("SLURM_CPUS_PER_TASK", 1) if "SLURM_JOB_ID" in os.environ else os.cpu_count() // 2
        req_workers = self.cfg.training.get("num_workers", -1)
        sys_worker = int(sys_worker)
        req_workers = int(req_workers)
        self.num_workers = min(sys_worker, req_workers) if req_workers > 0 else sys_worker

    def prepare_data(self):
        # Download or process data if needed
        pass

    def setup(self, stage=None):
        transforms = get_transforms(self.cfg)
        if self.cfg.task == "classification":
            self.train_dataset, self.val_dataset, self.test_dataset = get_dataset(self.cfg, transforms)
        elif self.cfg.task == "segmentation":
            self.train_dataset, self.val_dataset, self.test_dataset = get_dataset(self.cfg, transforms)
            if self.cfg.model.name.lower() == "mask2former":
                self.image_processor = Mask2FormerImageProcessor(
                    ignore_index=self.cfg.training.ignore_index,
                    reduce_labels=False,
                )
        elif self.cfg.task == "detection":
            self.train_dataset, self.val_dataset, self.test_dataset = get_dataset(
                self.cfg,
                transforms,
                bbox_format=self.cfg.model.bbox_format,
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

    def mask2former_collate_fn(self, batch):
        inputs = list(zip(*batch))
        images = inputs[0]
        segmentation_maps = inputs[1]

        processed_batch = self.image_processor(
            images,
            segmentation_maps=segmentation_maps,
            return_tensors="pt",
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
        )

        processed_batch["orig_image"] = inputs[2]
        processed_batch["orig_mask"] = inputs[3]
        return processed_batch

    def get_collate_fn(self):
        if self.cfg.task == "detection":
            if self.cfg.model.name.lower() == "efficientdet":
                return MarsDataModule.detection_collate_fn_v2
            else:
                return MarsDataModule.detection_collate_fn
        elif self.cfg.task == "segmentation":
            if self.cfg.model.name.lower() == "mask2former":
                return self.mask2former_collate_fn
            else:
                return None
        else:
            return None

    def train_dataloader(self):
        """Get train dataloader."""
        assert self.train_dataset is not None, "train_dataset is not loaded."
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.get_collate_fn(),
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        assert self.val_dataset is not None, "val_dataset is not loaded."
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.get_collate_fn(),
        )

    def test_dataloader(self):
        """Get test dataloader."""
        assert self.test_dataset is not None, "test_dataset is not loaded."
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.get_collate_fn(),
        )
