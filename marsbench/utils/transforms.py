"""
Image transformation utilities for data preprocessing and augmentation.
"""

import logging

from torchvision import transforms

logger = logging.getLogger(__name__)

# Map image types to PIL/Transform modes
IMAGE_MODES = {
    "rgb": "rgb",
    "grayscale": "grayscale",
    "l": "grayscale",
}


def get_geometric_transforms(image_size: tuple, is_train: bool = True):
    """Get geometric transformations (resize, flip, rotate) that should be applied to both image and mask."""
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply([transforms.RandomRotation(90)], p=0.25),
                transforms.RandomApply([transforms.RandomRotation(180)], p=0.25),
                transforms.RandomApply([transforms.RandomRotation(270)], p=0.25),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size),
        ]
    )


def get_image_transforms(cfg):
    """Get transforms for images with normalization from config."""
    image_size = tuple(cfg.transforms.image_size)
    # Get image mode from config
    requested_mode = cfg.data.image_type.lower().strip()
    image_mode = IMAGE_MODES.get(requested_mode)
    if image_mode is None:
        logger.error(
            f"Invalid/unsupported image_type '{requested_mode}'. Valid options are: {list(IMAGE_MODES.keys())}. "
            "Defaulting to RGB."
        )
        image_mode = "rgb"

    mean = cfg.transforms.get(image_mode).mean
    std = cfg.transforms.get(image_mode).std
    logger.info(f"Using {image_mode} normalization: mean={mean}, std={std}")

    # Get geometric transforms that will be shared with mask if needed
    train_geometric = get_geometric_transforms(image_size, is_train=True)
    val_geometric = get_geometric_transforms(image_size, is_train=False)

    train_transform = transforms.Compose(
        [
            train_geometric,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_transform = transforms.Compose(
        [
            val_geometric,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_transform, val_transform, train_geometric, val_geometric


def get_mask_transforms(geometric_transform):
    """Get transforms for segmentation masks, using the same geometric transforms as the image."""
    transform = transforms.Compose(
        [
            geometric_transform,
            transforms.ToTensor(),
        ]
    )

    return transform


def get_transforms(cfg):
    """Get appropriate transforms based on task."""
    (
        train_transform,
        val_transform,
        train_geometric,
        val_geometric,
    ) = get_image_transforms(cfg)

    if cfg.task == "segmentation":
        # Use training geometric transforms for training masks, validation for validation
        train_mask_transform = get_mask_transforms(train_geometric)
        val_mask_transform = get_mask_transforms(val_geometric)
        return train_transform, val_transform, train_mask_transform, val_mask_transform

    return train_transform, val_transform
