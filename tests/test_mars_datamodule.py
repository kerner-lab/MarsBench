import os

import pytest
import torch
from hydra import compose
from hydra import initialize_config_dir

from src.data.mars_datamodule import MarsDataModule


@pytest.mark.parametrize(
    "dataset_name,task",
    [
        ("HiRISENet", "classification"),
        ("ConeQuest", "segmentation"),
        ("conequest_detect", "detection"),
    ],
)
def test_mars_datamodule(dataset_name, task):
    # Load a sample configuration
    config_dir = os.path.abspath("configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name="config", overrides=[f"data={dataset_name}"])

    # Skip if dataset status is not ready
    if cfg.data.status not in cfg.test.data.status:
        print(f"Skipping dataset '{dataset_name}' (status: {cfg.data.status})")
        pytest.skip(
            f"Dataset '{dataset_name}' is not ready for testing (status: {cfg.data.status})"
        )

    # Initialize the MarsDataModule
    data_module = MarsDataModule(cfg)

    # Call setup to load datasets
    data_module.setup()

    # Verify that datasets are loaded
    assert (
        data_module.train_dataset is not None
    ), f"train_dataset for {dataset_name} is not loaded"
    assert (
        data_module.val_dataset is not None
    ), f"val_dataset for {dataset_name} is not loaded"
    assert (
        data_module.test_dataset is not None
    ), f"test_dataset for {dataset_name} is not loaded"

    # Verify dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    assert train_loader is not None, f"train_loader for {dataset_name} is not created"
    assert val_loader is not None, f"val_loader for {dataset_name} is not created"
    assert test_loader is not None, f"test_loader for {dataset_name} is not created"

    # Get a batch from each loader and verify them
    for split_name, batch in [
        ("train", next(iter(train_loader))),
        ("val", next(iter(val_loader))),
        ("test", next(iter(test_loader))),
    ]:
        # Check batch structure based on task
        if task == "classification":
            # For classification, we expect (images, labels)
            assert (
                len(batch) == 2
            ), f"{dataset_name} {split_name} batch should contain (images, labels)"
            images, labels = batch

            # Check image dimensions
            assert (
                len(images.shape) == 4
            ), f"{dataset_name} {split_name} images should be 4D (batch, channels, height, width)"
            batch_size, channels = images.shape[:2]
            assert channels == (
                1 if cfg.data.image_type == "grayscale" else 3
            ), f"{dataset_name} {split_name} should have {1 if cfg.data.image_type == 'grayscale' else 3} channels"

            # Check labels
            assert (
                labels.shape[0] == batch_size
            ), f"{dataset_name} {split_name} batch size mismatch between images and labels"
            assert (
                labels.dtype == torch.int64
            ), f"{dataset_name} {split_name} labels should be integers"
            assert (labels >= 0).all() and (
                labels < cfg.data.num_classes
            ).all(), f"{dataset_name} {split_name} labels should be in range [0, {cfg.data.num_classes})"

        elif task == "segmentation":
            # For segmentation, we expect (images, masks)
            assert (
                len(batch) == 2
            ), f"{dataset_name} {split_name} batch should contain (images, masks)"
            images, masks = batch

            # Check image dimensions
            assert (
                len(images.shape) == 4
            ), f"{dataset_name} {split_name} images should be 4D (batch, channels, height, width)"
            batch_size, channels = images.shape[:2]
            assert channels == (
                1 if cfg.data.image_type == "grayscale" else 3
            ), f"{dataset_name} {split_name} should have {1 if cfg.data.image_type == 'grayscale' else 3} channels"

            # Check masks
            assert (
                masks.shape[0] == batch_size
            ), f"{dataset_name} {split_name} batch size mismatch between images and masks"
            assert (
                masks.shape[2:] == images.shape[2:]
            ), f"{dataset_name} {split_name} spatial dimensions should match between images and masks"
            assert (masks >= 0).all() and (
                masks <= 1
            ).all(), (
                f"{dataset_name} {split_name} mask values should be in range [0, 1]"
            )

        elif task == "detection":
            # For detection, we expect (images, targets)
            assert (
                len(batch) == 2
            ), f"{dataset_name} {split_name} batch should contain (images, targets)"
            images, targets = batch

            # Check image dimensions
            assert (
                len(images.shape) == 4
            ), f"{dataset_name} {split_name} images should be 4D (batch, channels, height, width)"
            batch_size, channels = images.shape[:2]
            assert channels == (
                1 if cfg.data.image_type == "grayscale" else 3
            ), f"{dataset_name} {split_name} should have {1 if cfg.data.image_type == 'grayscale' else 3} channels"

            # Check targets
            assert isinstance(targets, tuple) and isinstance(
                targets[0], dict
            ), f"{dataset_name} {split_name} targets should be tuple of dictionaries"

            bbox_key = "boxes" if "boxes" in targets[0] else "bbox"
            class_key = "labels" if "labels" in targets[0] else "cls"

            for target in targets:
                assert isinstance(
                    target, dict
                ), f"{dataset_name} {split_name} targets should be a dictionary"

                # Check target keys
                assert (
                    bbox_key in target
                ), f"{dataset_name} {split_name} targets should contain '{bbox_key}'"
                assert (
                    class_key in target
                ), f"{dataset_name} {split_name} targets should contain '{class_key}'"

                bboxes = target[bbox_key]
                labels = target[class_key]

                # Check bounding boxes
                assert isinstance(
                    bboxes, torch.Tensor
                ), f"{dataset_name} {split_name} bboxes should be a tensor"
                assert (
                    bboxes.ndimension() == 2 and bboxes.shape[1] == 4
                ), f"{dataset_name} {split_name} bboxes should be 2D (N, 4)"

                # Check labels
                assert isinstance(
                    labels, torch.Tensor
                ), f"{dataset_name} {split_name} labels should be a tensor"
                assert (
                    labels.ndimension() == 1
                ), f"{dataset_name} {split_name} labels should be 1D (N,)"
                assert (
                    labels.shape[0] == bboxes.shape[0]
                ), f"{dataset_name} {split_name} labels should match number of bounding boxes"

        else:
            raise ValueError(f"Unknown task type: {task}")

        print(
            f"MarsDataModule for {dataset_name} ({task}) {split_name} split is working correctly"
        )


if __name__ == "__main__":
    pytest.main([__file__])
