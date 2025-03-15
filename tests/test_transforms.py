"""Tests for image transformation utilities using synthetic data."""

import logging
import os
from typing import Tuple

import numpy as np
import pytest
import torch
from hydra import compose
from hydra import initialize_config_dir
from omegaconf import DictConfig
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from src.utils.transforms import get_geometric_transforms
from src.utils.transforms import get_transforms


def create_random_image(size: Tuple[int, int], channels: int = 3) -> Image.Image:
    """Create a random PIL image for testing transforms."""
    # Create random tensor of appropriate shape
    if channels == 1:
        # For grayscale images
        tensor = torch.rand(size)
        # Convert to PIL Image (L mode)
        return Image.fromarray((tensor * 255).byte().numpy(), mode="L")
    else:
        # For RGB images
        tensor = torch.rand(size[0], size[1], channels)
        # Convert to PIL Image
        return Image.fromarray((tensor * 255).byte().numpy().astype("uint8"), mode="RGB")


def create_random_mask(size: Tuple[int, int]) -> Image.Image:
    """Create a random binary mask for testing segmentation transforms."""
    # Create random binary tensor
    tensor = (torch.rand(size) > 0.5).float()
    # Convert to PIL Image (L mode)
    return Image.fromarray((tensor * 255).byte().numpy(), mode="L")


@pytest.fixture
def sample_config() -> DictConfig:
    """Create a sample configuration for testing transforms."""
    config_dir = os.path.abspath("configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        # First try classification config
        try:
            return compose(
                config_name="config",
                overrides=["task=classification", "data=classification/domars16k"],
            )
        except Exception as e:
            logging.warning(f"Could not load classification config: {e}")
            # Fall back to a basic config if needed
            return DictConfig(
                {
                    "task": "classification",
                    "transforms": {
                        "image_size": [224, 224],
                        "rgb": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                        "grayscale": {"mean": [0.5], "std": [0.5]},
                    },
                    "data": {
                        "image_type": "rgb",
                    },
                }
            )


@pytest.mark.parametrize("task", ["classification", "segmentation"])
@pytest.mark.parametrize("image_type", ["rgb", "grayscale"])
def test_get_transforms(sample_config: DictConfig, task: str, image_type: str):
    """Test that get_transforms returns the correct number of transforms for each task."""
    # Update config for this test
    sample_config.task = task
    sample_config.data.image_type = image_type

    # Get the transforms
    transforms = get_transforms(sample_config)

    # Check number of transforms based on task
    if task == "classification":
        assert len(transforms) == 2, f"Expected 2 transforms for {task}, got {len(transforms)}"
        train_transform, val_transform = transforms
    else:  # segmentation
        assert len(transforms) == 4, f"Expected 4 transforms for {task}, got {len(transforms)}"
        train_transform, val_transform, train_mask_transform, val_mask_transform = transforms

    # Create random test image
    channels = 1 if image_type == "grayscale" else 3
    image_size = tuple(sample_config.transforms.image_size)
    test_image = create_random_image(image_size, channels)

    # Test image transforms
    train_output = train_transform(test_image)
    val_output = val_transform(test_image)

    # Check output shapes
    expected_channels = 1 if image_type == "grayscale" else 3
    msg = f"Expected {expected_channels} channels, got "
    assert train_output.shape[0] == expected_channels, f"{msg}{train_output.shape[0]}"
    assert val_output.shape[0] == expected_channels, f"{msg}{val_output.shape[0]}"

    expected_size = torch.Size(image_size)
    assert train_output.shape[1:] == expected_size, f"Expected size {image_size}, got {train_output.shape[1:]}"
    assert val_output.shape[1:] == expected_size, f"Expected size {image_size}, got {val_output.shape[1:]}"

    # Test mask transforms for segmentation
    if task == "segmentation":
        test_mask = create_random_mask(image_size)
        train_mask_output = train_mask_transform(test_mask)
        val_mask_output = val_mask_transform(test_mask)

        # Check mask output shapes and properties
        assert train_mask_output.shape[0] == 1, f"Expected 1 channel for mask, got {train_mask_output.shape[0]}"
        assert val_mask_output.shape[0] == 1, f"Expected 1 channel for mask, got {val_mask_output.shape[0]}"
        expected_shape = torch.Size(image_size)
        assert train_mask_output.shape[1:] == expected_shape, "Mask shape mismatch"
        assert val_mask_output.shape[1:] == expected_shape, "Mask shape mismatch"

        # Check mask values are binary (0 or 1)
        assert torch.all((train_mask_output == 0) | (train_mask_output == 1)), "Mask values should be binary"
        assert torch.all((val_mask_output == 0) | (val_mask_output == 1)), "Mask values should be binary"


def test_geometric_transforms():
    """Test geometric transforms."""
    # Testing train transform
    transform = get_geometric_transforms(image_size=(224, 224), is_train=True)
    assert transform is not None
    assert isinstance(transform, transforms.Compose)

    # Testing val transform
    transform = get_geometric_transforms(image_size=(224, 224), is_train=False)
    assert transform is not None
    assert isinstance(transform, transforms.Compose)


def test_transforms_segmentation():
    """Test transforms for segmentation."""
    # Dummy cfg
    cfg = OmegaConf.create(
        {
            "task": "segmentation",
            "image_size": [224, 224],
            "transform": {"mask_size": [224, 224]},
        }
    )

    train_transform, val_transform = get_transforms(cfg)
    assert train_transform is not None
    assert val_transform is not None

    # Test transforms with dummy image and mask
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (224, 224), dtype=np.uint8)

    transformed = train_transform(image=img, mask=mask)
    assert "image" in transformed
    assert "mask" in transformed
    assert transformed["image"].shape == (3, 224, 224)
    assert transformed["mask"].shape == (224, 224)

    transformed = val_transform(image=img, mask=mask)
    assert "image" in transformed
    assert "mask" in transformed
    assert transformed["image"].shape == (3, 224, 224)
    assert transformed["mask"].shape == (224, 224)

    # Test PIL Image transformation
    img_pil = Image.fromarray(img)
    mask_pil = Image.fromarray(mask)

    transformed = train_transform(image=img_pil, mask=mask_pil)
    assert "image" in transformed
    assert "mask" in transformed
    assert transformed["image"].shape == (3, 224, 224)
    assert transformed["mask"].shape == (224, 224)

    transformed = val_transform(image=img_pil, mask=mask_pil)
    assert "image" in transformed
    assert "mask" in transformed
    assert transformed["image"].shape == (3, 224, 224)
    assert transformed["mask"].shape == (224, 224)


def test_transforms_classification():
    """Test transforms for classification."""
    # Dummy cfg
    cfg = OmegaConf.create({"task": "classification", "image_size": [224, 224], "transform": {}})

    train_transform, val_transform = get_transforms(cfg)
    assert train_transform is not None
    assert val_transform is not None

    # Test transforms with dummy image
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    transformed = train_transform(image=img)
    assert "image" in transformed
    assert transformed["image"].shape == (3, 224, 224)

    transformed = val_transform(image=img)
    assert "image" in transformed
    assert transformed["image"].shape == (3, 224, 224)

    # Test PIL Image transformation
    img_pil = Image.fromarray(img)

    transformed = train_transform(image=img_pil)
    assert "image" in transformed
    assert transformed["image"].shape == (3, 224, 224)

    transformed = val_transform(image=img_pil)
    assert "image" in transformed
    assert transformed["image"].shape == (3, 224, 224)


if __name__ == "__main__":
    pytest.main([__file__])
