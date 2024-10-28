import torch
from torch.utils.data import random_split, Dataset, Subset
from .classification import (
    DoMars16k,
    HiRISENet,
    MSLNet,
    DeepMars_Landmark,
    DeepMars_Surface,
    MartianFrost,
)
from typing import Tuple, Union

def get_dataset(cfg, train_transform=None, val_transform=None, subset: Union[int, None]=None) -> Tuple[Dataset, Dataset, Dataset]:
    dataset_name = cfg.data.name
    if dataset_name == 'DoMars16k':
        train_dataset = DoMars16k(
            cfg = cfg,
            data_dir=cfg.data.data_dir.train,
            transform=train_transform,
        )
        val_dataset = DoMars16k(
            cfg = cfg,
            data_dir=cfg.data.data_dir.val,
            transform=val_transform,
        )
        test_dataset = DoMars16k(
            cfg = cfg,
            data_dir=cfg.data.data_dir.test,
            transform=val_transform,
        )
    elif dataset_name == 'HiRISENet':
        train_dataset = HiRISENet(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=train_transform,
            txt_file=cfg.data.txt_file,
            split_type="train"
        )
        val_dataset = HiRISENet(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            txt_file=cfg.data.txt_file,
            split_type="val"
        )
        test_dataset = HiRISENet(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            txt_file=cfg.data.txt_file,
            split_type="test"
        )
    elif dataset_name == 'MSLNet':
        train_dataset = MSLNet(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=train_transform,
            txt_file=cfg.data.txt_files.train
        )
        val_dataset = MSLNet(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            txt_file=cfg.data.txt_files.val
        )
        test_dataset = MSLNet(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            txt_file=cfg.data.txt_files.test
        )
    elif dataset_name == 'DeepMars_Landmark':
        full_dataset = DeepMars_Landmark(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=train_transform,
            txt_file=cfg.data.txt_file
        )
        total_size = len(full_dataset)
        train_size = int(0.6 * total_size)
        val_size = int(0.2 * total_size)
        test_size = total_size - train_size - val_size
        # Get indices for the splits
        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset, test_subset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )

        # Extract indices from the subsets
        train_indices = train_subset.indices
        val_indices = val_subset.indices
        test_indices = test_subset.indices

        # Create separate datasets for each split with their own transforms
        train_dataset = DeepMars_Landmark(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=train_transform,
            txt_file=cfg.data.txt_file,
            indices=train_indices
        )
        val_dataset = DeepMars_Landmark(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            txt_file=cfg.data.txt_file,
            indices=val_indices
        )
        test_dataset = DeepMars_Landmark(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            txt_file=cfg.data.txt_file,
            indices=test_indices
        )
    elif dataset_name == 'DeepMars_Surface':
        train_dataset = DeepMars_Surface(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=train_transform,
            txt_file=cfg.data.txt_files.train
        )
        val_dataset = DeepMars_Surface(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            txt_file=cfg.data.txt_files.val
        )
        test_dataset = DeepMars_Surface(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            txt_file=cfg.data.txt_files.test
        )
    elif dataset_name == 'MartianFrost':
        train_dataset = MartianFrost(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=train_transform,
            txt_file=cfg.data.txt_files.train
        )
        val_dataset = MartianFrost(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            txt_file=cfg.data.txt_files.val
        )
        test_dataset = MartianFrost(
            cfg=cfg,
            data_dir=cfg.data.data_dir,
            transform=val_transform,
            txt_file=cfg.data.txt_files.test
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized.")
    if subset is not None:
        train_dataset = Subset(train_dataset, range(subset))
        val_dataset = Subset(val_dataset, range(subset))
        test_dataset = Subset(test_dataset, range(subset))
    return train_dataset, val_dataset, test_dataset
