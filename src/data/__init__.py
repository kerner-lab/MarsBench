from typing import Tuple
from typing import Union

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch.utils.data import Subset

from .classification import DeepMars_Landmark
from .classification import DeepMars_Surface
from .classification import DoMars16k
from .classification import HiRISENet
from .classification import MartianFrost
from .classification import MSLNet


def get_dataset(
    cfg: DictConfig,
    train_transform: torch.nn.Module,
    val_transform: torch.nn.Module,
    subset: Union[int, None] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    dataset_name = cfg.data.name
    if dataset_name == "DoMars16k":
        train_dataset = DoMars16k(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=train_transform,
            annot_csv=cfg.data.annot_csv,
        )
        val_dataset = DoMars16k(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            annot_csv=cfg.data.annot_csv,
        )
        test_dataset = DoMars16k(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            annot_csv=cfg.data.annot_csv,
        )
    elif dataset_name == "HiRISENet":
        train_dataset = HiRISENet(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=train_transform,
            annot_csv=cfg.data.annot_csv,
            split="train",
        )
        val_dataset = HiRISENet(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            annot_csv=cfg.data.annot_csv,
            split="val",
        )
        test_dataset = HiRISENet(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            annot_csv=cfg.data.annot_csv,
            split="test",
        )
    elif dataset_name == "MSLNet":
        train_dataset = MSLNet(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=train_transform,
            annot_csv=cfg.data.annot_csv,
            split="train",
        )
        val_dataset = MSLNet(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            annot_csv=cfg.data.annot_csv,
            split="val",
        )
        test_dataset = MSLNet(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            annot_csv=cfg.data.annot_csv,
            split="test",
        )
    elif dataset_name == "DeepMars_Landmark":
        train_dataset = DeepMars_Landmark(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=train_transform,
            annot_csv=cfg.data.annot_csv,
            split="train",
        )
        val_dataset = DeepMars_Landmark(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            annot_csv=cfg.data.annot_csv,
            split="val",
        )
        test_dataset = DeepMars_Landmark(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            annot_csv=cfg.data.annot_csv,
            split="test",
        )
    elif dataset_name == "DeepMars_Surface":
        train_dataset = DeepMars_Surface(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=train_transform,
            annot_csv=cfg.data.annot_csv,
            split="train",
        )
        val_dataset = DeepMars_Surface(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            annot_csv=cfg.data.annot_csv,
            split="val",
        )
        test_dataset = DeepMars_Surface(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            annot_csv=cfg.data.annot_csv,
            split="test",
        )
    elif dataset_name == "MartianFrost":
        train_dataset = MartianFrost(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=train_transform,
            annot_csv=cfg.data.annot_csv,
            split="train",
        )
        val_dataset = MartianFrost(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            annot_csv=cfg.data.annot_csv,
            split="val",
        )
        test_dataset = MartianFrost(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            annot_csv=cfg.data.annot_csv,
            split="test",
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized.")
    if subset is not None:
        train_dataset = Subset(train_dataset, range(subset))
        val_dataset = Subset(val_dataset, range(subset))
        test_dataset = Subset(test_dataset, range(subset))
    return train_dataset, val_dataset, test_dataset
