"""
Dataset loading and preprocessing utilities for MarsBench.
"""

import logging
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch.utils.data import Subset

from marsbench.data.segmentation.Mask2FormerWrapper import Mask2FormerWrapper

from .classification import Atmospheric_Dust_Classification_EDR
from .classification import Atmospheric_Dust_Classification_RDR
from .classification import Change_Classification_CTX
from .classification import Change_Classification_HiRISE
from .classification import DoMars16k
from .classification import Frost_Classification
from .classification import Landmark_Classification
from .classification import Multi_Label_MER
from .classification import Surface_Classification
from .detection import Boulder_Detection
from .detection import ConeQuest_Detection
from .detection import Dust_Devil_Detection
from .segmentation import MMLS
from .segmentation import Boulder_Segmentation
from .segmentation import ConeQuest_Segmentation
from .segmentation import Crater_Binary_Segmentation
from .segmentation import Crater_Multi_Segmentation
from .segmentation import MarsSegMER
from .segmentation import MarsSegMSL
from .segmentation import S5Mars

logger = logging.getLogger(__name__)

DATASET_REGISTRY = {
    "classification": {
        "DoMars16k": DoMars16k,
        "Landmark_Classification": Landmark_Classification,
        "Surface_Classification": Surface_Classification,
        "Frost_Classification": Frost_Classification,
        "Atmospheric_Dust_Classification_RDR": Atmospheric_Dust_Classification_RDR,
        "Atmospheric_Dust_Classification_EDR": Atmospheric_Dust_Classification_EDR,
        "Change_Classification_HiRISE": Change_Classification_HiRISE,
        "Change_Classification_CTX": Change_Classification_CTX,
        "Multi_Label_MER": Multi_Label_MER,
    },
    "segmentation": {
        "ConeQuest_Segmentation": ConeQuest_Segmentation,
        "Boulder_Segmentation": Boulder_Segmentation,
        "MarsSegMER": MarsSegMER,
        "MarsSegMSL": MarsSegMSL,
        "MMLS": MMLS,
        "S5Mars": S5Mars,
        "Crater_Binary_Segmentation": Crater_Binary_Segmentation,
        "Crater_Multi_Segmentation": Crater_Multi_Segmentation,
    },
    "detection": {
        "ConeQuest_Detection": ConeQuest_Detection,
        "Dust_Devil_Detection": Dust_Devil_Detection,
        "Boulder_Detection": Boulder_Detection,
    },
}


def instantiate_dataset(dataset_class, cfg, transform, split, bbox_format=None, annot_csv=None):
    common_args = {
        "cfg": cfg,
        "data_dir": cfg.data.data_dir,
        "transform": transform,
        "split": split,
    }
    partition_support = ["classification", "segmentation"]
    few_shot_support = ["classification"]
    if (cfg.partition is not None and cfg.task in partition_support) or (
        cfg.few_shot is not None and cfg.task in few_shot_support
    ):
        if cfg.few_shot is not None and cfg.partition is not None:
            msg = "At most one of cfg.few_shot or cfg.partition may be set; both cannot be non-None"
            logger.error(msg)
            raise ValueError(msg)
        if cfg.few_shot is not None:
            annot_csv = cfg.data.few_shot_csv
        elif cfg.partition is not None:
            annot_csv = cfg.data.partition_csv
            if cfg.partition >= 0.1:
                annot_csv_partition = annot_csv.split("/")[-1]
                annot_csv_partition_val = annot_csv_partition.split("x_")[0]
                new_val = f"{cfg.partition:.2f}"
                logger.info(f"Old annotation csv path is: {annot_csv}")
                annot_csv = annot_csv.replace(f"{annot_csv_partition_val}x_", f"{new_val}x_")
                logger.info(f"Updated annotation csv path is: {annot_csv}")
    elif cfg.partition is not None:
        logger.warning(f"Task: {cfg.task} does not support partition. Using whole data.")
    elif cfg.few_shot is not None:
        logger.warning(f"Task: {cfg.task} does not support few shot. Using whole data.")

    if cfg.task in ["classification", "segmentation"]:
        common_args["annot_csv"] = cfg.data.annot_csv if annot_csv is None else annot_csv
    if cfg.task == "detection":
        common_args["bbox_format"] = bbox_format
    return dataset_class(**common_args)


def get_dataset(
    cfg: DictConfig,
    transforms: Tuple[torch.nn.Module, torch.nn.Module],
    subset: Union[int, None] = None,
    bbox_format: Optional[str] = None,
    annot_csv: Optional[str] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Returns a train, val, and test dataset.

    Args:
        cfg (DictConfig):
            Configuration dictionary.
        transforms (Tuple[torch.nn.Module, torch.nn.Module]):
            Tuple of train and val transforms.
        subset (Union[int, None], optional):
            Number of samples to use for training. Prioritizes cfg.data.subset over this argument.
        annot_csv (Optional[str], optional):
            Path to annotation CSV file.
    Returns:
        Tuple[Dataset, Dataset, Dataset]: Tuple of train, val, and test datasets.
    """

    try:
        dataset_cls = DATASET_REGISTRY[cfg.task][cfg.data.name]
    except KeyError as e:
        logger.error(f"Unsupported dataset {cfg.data.name} for task {cfg.task}")
        logger.debug(f"Available datasets for task {cfg.task}: {DATASET_REGISTRY[cfg.task]}")
        raise ValueError(f"Dataset not supported: {cfg.data.name} for {cfg.task}") from e

    train_dataset = instantiate_dataset(dataset_cls, cfg, transforms[0], "train", bbox_format, annot_csv)
    val_dataset = instantiate_dataset(dataset_cls, cfg, transforms[1], "val", bbox_format, annot_csv)
    test_dataset = instantiate_dataset(dataset_cls, cfg, transforms[1], "test", bbox_format, annot_csv)

    # Dataset wrapper for Mask2Former
    if cfg.model.name.lower() == "mask2former":
        train_dataset = Mask2FormerWrapper(train_dataset)
        val_dataset = Mask2FormerWrapper(val_dataset)
        test_dataset = Mask2FormerWrapper(test_dataset)

    # Apply subset if specified (prioritizing cfg.data.subset)
    actual_subset = cfg.data.get("subset", None) or subset
    if actual_subset is not None and actual_subset > 0:
        indices = torch.randperm(len(train_dataset))[:actual_subset].tolist()
        train_dataset = Subset(train_dataset, indices)

    return train_dataset, val_dataset, test_dataset

    # # Classification datasets
    # if cfg.task == "classification":
    #     if cfg.data.name == "DoMars16k":
    #         train_dataset = DoMars16k(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[0],
    #             annot_csv=cfg.data.annot_csv,
    #             split="train",
    #         )
    #         val_dataset = DoMars16k(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[1],
    #             annot_csv=cfg.data.annot_csv,
    #             split="val",
    #         )
    #         test_dataset = DoMars16k(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[1],
    #             annot_csv=cfg.data.annot_csv,
    #             split="test",
    #         )
    #     elif cfg.data.name == "DeepMars_Landmark":
    #         train_dataset = DeepMars_Landmark(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[0],
    #             annot_csv=cfg.data.annot_csv,
    #             split="train",
    #         )
    #         val_dataset = DeepMars_Landmark(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[1],
    #             annot_csv=cfg.data.annot_csv,
    #             split="val",
    #         )
    #         test_dataset = DeepMars_Landmark(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[1],
    #             annot_csv=cfg.data.annot_csv,
    #             split="test",
    #         )
    #     elif cfg.data.name == "DeepMars_Surface":
    #         train_dataset = DeepMars_Surface(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[0],
    #             annot_csv=cfg.data.annot_csv,
    #             split="train",
    #         )
    #         val_dataset = DeepMars_Surface(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[1],
    #             annot_csv=cfg.data.annot_csv,
    #             split="val",
    #         )
    #         test_dataset = DeepMars_Surface(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[1],
    #             annot_csv=cfg.data.annot_csv,
    #             split="test",
    #         )
    #     elif cfg.data.name == "HiRISENet":
    #         train_dataset = HiRISENet(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[0],
    #             annot_csv=cfg.data.annot_csv,
    #             split="train",
    #         )
    #         val_dataset = HiRISENet(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[1],
    #             annot_csv=cfg.data.annot_csv,
    #             split="val",
    #         )
    #         test_dataset = HiRISENet(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[1],
    #             annot_csv=cfg.data.annot_csv,
    #             split="test",
    #         )
    #     elif cfg.data.name == "MartianFrost":
    #         train_dataset = MartianFrost(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[0],
    #             annot_csv=cfg.data.annot_csv,
    #             split="train",
    #         )
    #         val_dataset = MartianFrost(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[1],
    #             annot_csv=cfg.data.annot_csv,
    #             split="val",
    #         )
    #         test_dataset = MartianFrost(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[1],
    #             annot_csv=cfg.data.annot_csv,
    #             split="test",
    #         )
    #     elif cfg.data.name == "MSLNet":
    #         train_dataset = MSLNet(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[0],
    #             annot_csv=cfg.data.annot_csv,
    #             split="train",
    #         )
    #         val_dataset = MSLNet(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[1],
    #             annot_csv=cfg.data.annot_csv,
    #             split="val",
    #         )
    #         test_dataset = MSLNet(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[1],
    #             annot_csv=cfg.data.annot_csv,
    #             split="test",
    #         )
    #     else:
    #         raise ValueError(f"Dataset not supported: {cfg.data.name} for {cfg.task}")

    # # Segmentation datasets
    # elif cfg.task == "segmentation":
    #     if cfg.data.name == "ConeQuest":
    #         train_dataset = ConeQuest(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[0],
    #             split="train",
    #         )
    #         val_dataset = ConeQuest(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[1],
    #             split="val",
    #         )
    #         test_dataset = ConeQuest(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[1],
    #             split="test",
    #         )
    #     else:
    #         raise ValueError(f"Dataset not supported: {cfg.data.name} for {cfg.task}")

    # # Detection datasets
    # elif cfg.task == "detection":
    #     if cfg.data.name == "ConeQuest":
    #         train_dataset = ConeQuestDetection(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[0],
    #             bbox_format=bbox_format,
    #             split="train",
    #         )
    #         val_dataset = ConeQuestDetection(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[1],
    #             bbox_format=bbox_format,
    #             split="val",
    #         )
    #         test_dataset = ConeQuestDetection(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[1],
    #             bbox_format=bbox_format,
    #             split="test",
    #         )
    #     elif cfg.data.name == "Mars_Dust_Devil":
    #         train_dataset = Mars_Dust_Devil(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[0],
    #             bbox_format=bbox_format,
    #             split="train",
    #         )
    #         val_dataset = Mars_Dust_Devil(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[1],
    #             bbox_format=bbox_format,
    #             split="val",
    #         )
    #         test_dataset = Mars_Dust_Devil(
    #             cfg=cfg,
    #             data_dir=cfg.data.data_dir,
    #             transform=transforms[1],
    #             bbox_format=bbox_format,
    #             split="test",
    #         )
    #     else:
    #         raise ValueError(f"Dataset not supported: {cfg.data.name} for {cfg.task}")

    # else:
    #     raise ValueError(f"Task not supported: {cfg.task}")

    # # Apply subset if specified
    # subset = cfg.data.subset if cfg.data.get("subset", None) is not None else subset
    # if subset is not None and subset > 0:
    #     train_dataset = Subset(
    #         train_dataset,
    #         torch.randperm(len(train_dataset))[:subset].tolist(),
    #     )

    # return train_dataset, val_dataset, test_dataset
