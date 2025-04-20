"""
Image transformation utilities for data preprocessing and augmentation.

This module provides a unified approach for creating transforms for different tasks
(classification, segmentation, detection) using the albumentations library.
"""

import logging

import albumentations as A
import cv2

logger = logging.getLogger(__name__)

# Map image types to appropriate modes
IMAGE_MODES = {
    "rgb": "rgb",
    "grayscale": "grayscale",
    "l": "grayscale",
}


def get_transforms(cfg):
    """Get appropriate transforms based on task.

    This function serves as the main entry point for creating transforms
    for any task in the MarsBench pipeline.

    Args:
        cfg (DictConfig): Configuration dictionary

    Returns:
        tuple: (train_transform, val_transform) where each is an appropriate transform
               for the specified task

    Raises:
        ValueError: If the task specified in cfg is not supported
    """
    # Get image size from config
    image_size = (
        tuple(cfg.model.input_size)[1:] if cfg.model.get("input_size", None) else tuple(cfg.transforms.image_size)
    )

    # Get image mode and normalization parameters
    requested_mode = cfg.data.image_type.lower().strip()
    image_mode = IMAGE_MODES.get(requested_mode)
    if image_mode is None:
        logger.error(
            f"Invalid/unsupported image_type '{requested_mode}'. Valid options are: {list(IMAGE_MODES.keys())}. "
            "Defaulting to RGB."
        )
        image_mode = "rgb"

    mean = tuple(cfg.transforms.get(image_mode).mean)
    std = tuple(cfg.transforms.get(image_mode).std)
    logger.info(f"Using {image_mode} normalization: mean={mean}, std={std}")

    # Create task-specific transforms
    if cfg.task == "segmentation":
        return _get_segmentation_transforms(image_size, mean, std)
    elif cfg.task == "classification":
        return _get_classification_transforms(image_size, mean, std)
    elif cfg.task == "detection":
        return _get_detection_transforms(cfg, image_size, mean, std)
    else:
        logger.error(
            f"Unsupported task type: '{cfg.task}'. Supported tasks are: classification, segmentation, detection"
        )
        raise ValueError(f"Unsupported task type: '{cfg.task}'")


def _get_classification_transforms(image_size, mean, std):
    """Create transforms for classification tasks."""
    train_transform = A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.GaussianBlur(p=0.2),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            A.ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            A.ToTensorV2(),
        ]
    )

    return train_transform, val_transform


def _get_segmentation_transforms(image_size, mean, std):
    """Create transforms for segmentation tasks."""
    train_transform = A.Compose(
        [
            A.Resize(
                height=image_size[0],
                width=image_size[1],
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            A.ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )

    val_transform = A.Compose(
        [
            A.Resize(
                height=image_size[0],
                width=image_size[1],
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            A.ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )

    return train_transform, val_transform


def _get_detection_transforms(cfg, image_size, mean, std):
    """Create transforms for object detection tasks."""
    bbox_format = cfg.model.bbox_format

    train_transform = A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1], p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            A.ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format=bbox_format, label_fields=["class_labels"]),
    )

    val_transform = A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1], p=1.0),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            A.ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format=bbox_format, label_fields=["class_labels"]),
    )

    return train_transform, val_transform
