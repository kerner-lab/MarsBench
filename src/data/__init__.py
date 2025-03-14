import logging
from typing import Optional
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
from .segmentation import ConeQuest

logger = logging.getLogger(__name__)


def get_dataset(
    cfg: DictConfig,
    transforms: Tuple[torch.nn.Module, torch.nn.Module],
    subset: Union[int, None] = None,
    mask_transforms: Optional[Tuple[torch.nn.Module, torch.nn.Module]] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Returns a train, val, and test dataset.

    Args:
        cfg (DictConfig):
            Configuration dictionary.
        transforms (Tuple[torch.nn.Module, torch.nn.Module]):
            Tuple of train and val transforms.
        subset (Union[int, None], optional):
            Number of samples to use for training. Defaults to None.
        mask_transforms (Optional[Tuple[torch.nn.Module, torch.nn.Module]], optional):
            Tuple of train and val mask transforms. Defaults to None.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: Tuple of train, val, and test datasets.
    """

    # Classification datasets
    if cfg.task == "classification":
        if cfg.data.name == "DoMars16k":
            train_dataset = DoMars16k(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[0],
                annot_csv=cfg.data.annot_csv,
                split="train",
            )
            val_dataset = DoMars16k(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[1],
                annot_csv=cfg.data.annot_csv,
                split="val",
            )
            test_dataset = DoMars16k(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[1],
                annot_csv=cfg.data.annot_csv,
                split="test",
            )
        elif cfg.data.name == "DeepMars_Landmark":
            train_dataset = DeepMars_Landmark(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[0],
                annot_csv=cfg.data.annot_csv,
                split="train",
            )
            val_dataset = DeepMars_Landmark(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[1],
                annot_csv=cfg.data.annot_csv,
                split="val",
            )
            test_dataset = DeepMars_Landmark(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[1],
                annot_csv=cfg.data.annot_csv,
                split="test",
            )
        elif cfg.data.name == "DeepMars_Surface":
            train_dataset = DeepMars_Surface(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[0],
                annot_csv=cfg.data.annot_csv,
                split="train",
            )
            val_dataset = DeepMars_Surface(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[1],
                annot_csv=cfg.data.annot_csv,
                split="val",
            )
            test_dataset = DeepMars_Surface(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[1],
                annot_csv=cfg.data.annot_csv,
                split="test",
            )
        elif cfg.data.name == "HiRISENet":
            train_dataset = HiRISENet(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[0],
                annot_csv=cfg.data.annot_csv,
                split="train",
            )
            val_dataset = HiRISENet(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[1],
                annot_csv=cfg.data.annot_csv,
                split="val",
            )
            test_dataset = HiRISENet(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[1],
                annot_csv=cfg.data.annot_csv,
                split="test",
            )
        elif cfg.data.name == "MartianFrost":
            train_dataset = MartianFrost(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[0],
                annot_csv=cfg.data.annot_csv,
                split="train",
            )
            val_dataset = MartianFrost(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[1],
                annot_csv=cfg.data.annot_csv,
                split="val",
            )
            test_dataset = MartianFrost(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[1],
                annot_csv=cfg.data.annot_csv,
                split="test",
            )
        elif cfg.data.name == "MSLNet":
            train_dataset = MSLNet(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[0],
                annot_csv=cfg.data.annot_csv,
                split="train",
            )
            val_dataset = MSLNet(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[1],
                annot_csv=cfg.data.annot_csv,
                split="val",
            )
            test_dataset = MSLNet(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[1],
                annot_csv=cfg.data.annot_csv,
                split="test",
            )
        else:
            raise ValueError(f"Dataset not supported: {cfg.data.name} for {cfg.task}")

    # Segmentation datasets
    elif cfg.task == "segmentation":
        if cfg.data.name == "ConeQuest":
            if mask_transforms is None:
                raise ValueError("mask_transforms required for segmentation datasets")

            train_dataset = ConeQuest(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[0],
                mask_transform=mask_transforms[0],
                split="train",
            )
            val_dataset = ConeQuest(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[1],
                mask_transform=mask_transforms[1],
                split="val",
            )
            test_dataset = ConeQuest(
                cfg=cfg,
                data_dir=cfg.data.data_dir,
                transform=transforms[1],
                mask_transform=mask_transforms[1],
                split="test",
            )
        else:
            raise ValueError(f"Dataset not supported: {cfg.data.name} for {cfg.task}")

    else:
        raise ValueError(f"Task not supported: {cfg.task}")

    if subset is not None and subset > 0:
        train_dataset = Subset(
            train_dataset,
            torch.randperm(len(train_dataset))[:subset].tolist(),
        )

    return train_dataset, val_dataset, test_dataset
