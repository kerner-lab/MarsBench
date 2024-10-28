import logging

import numpy as np
import torch
import wandb
from torch.utils.data import Subset


# Used for loading text files for Martian Frost
def load_text_ids(file_path):
    """Helper to load all lines from a text file"""
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


# Used for splitting Martian Frost dataset into subsequent train val test datasets
def split_dataset(dataset, train_ids, validate_ids, test_ids):
    train_indices = []
    validate_indices = []
    test_indices = []

    for idx, subframe_id in enumerate(dataset.subframe_ids):
        if subframe_id in train_ids:
            train_indices.append(idx)
        elif subframe_id in validate_ids:
            validate_indices.append(idx)
        elif subframe_id in test_ids:
            test_indices.append(idx)
        else:
            logging.warning(
                f"{subframe_id}: Did not find designated split in train/validate/test list."
            )

    return (
        Subset(dataset, train_indices),
        Subset(dataset, validate_indices),
        Subset(dataset, test_indices),
    )


# Bootstrap sampler if required
def bootstrap_sampler(dataset, num_samples):
    ind = np.random.choice(len(dataset), size=num_samples, replace=True)
    return torch.utils.data.Subset(dataset, ind)


def uniform_sampler(train_dataset):
    # Creating weights for uniform sampling
    label_count = np.bincount(train_dataset.labels)
    inverse_weights = 1.0 / label_count
    sample_weights = inverse_weights[train_dataset.labels]

    # Uniform sampler on the basis of inverse counts
    uniform_sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    return uniform_sampler


# Image logger if required
def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    table = wandb.Table(
        columns=["image", "pred", "target"] + [f"score_{i}" for i in range(20)]
    )
    for img, pred, targ, prob in zip(
        images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")
    ):
        table.add_data(wandb.Image(img[0].numpy() * 255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table": table}, commit=False)
