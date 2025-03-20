import glob
import os

import pytest
import torch
from hydra import compose
from hydra import initialize_config_dir

from marsbench.data import get_dataset
from marsbench.utils.transforms import get_transforms
from tests.conftest import skip_if_ci
from tests.utils.dataset_test_utils import check_bboxes_coco
from tests.utils.dataset_test_utils import check_bboxes_pascal_voc
from tests.utils.dataset_test_utils import check_bboxes_yolo


def initialize_datasets(cfg, transforms, bbox_format=None):
    if cfg.task == "classification":
        train_dataset, val_dataset, test_dataset = get_dataset(cfg, transforms[:2], subset=cfg.test.data.subset_size)
    elif cfg.task == "segmentation":
        train_dataset, val_dataset, test_dataset = get_dataset(
            cfg,
            transforms[:2],
            subset=cfg.test.data.subset_size,
            mask_transforms=transforms[2:],
        )
    elif cfg.task == "detection":
        train_dataset, val_dataset, test_dataset = get_dataset(
            cfg,
            transforms[:2],
            subset=cfg.test.data.subset_size,
            bbox_format=bbox_format,
        )
    else:
        raise ValueError(f"Task not yet supported: {cfg.task}")

    return train_dataset, val_dataset, test_dataset


@skip_if_ci
@pytest.mark.parametrize("dataset_config_file", glob.glob("configs/data/**/*.yaml"))
def test_datasets(dataset_config_file):
    # Get config directory
    config_dir = os.path.abspath("configs")

    # Initialize Hydra and compose configuration
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        # Get dataset path relative to configs/data
        rel_path = os.path.relpath(dataset_config_file, os.path.join(config_dir, "data"))
        # Remove .yaml extension
        rel_path = os.path.splitext(rel_path)[0]

        # Extract task from the path
        task = rel_path.split("/")[0]

        cfg = compose(
            config_name="config",
            overrides=[f"data={rel_path}", f"task={task}", "+model.bbox_format=null"],
        )

        # Check dataset status
        if cfg.data.status not in cfg.test.data.status:
            print(f"Skipping dataset '{cfg.data.name}' (status: {cfg.data.status})")
            pytest.skip(f"Dataset '{cfg.data.name}' is not in status set {cfg.test.data.status} for testing.")

        # Check if dataset files exist
        data_dir = cfg.data.data_dir
        annot_csv = cfg.data.annot_csv if task != "detection" else None
        if not os.path.exists(data_dir) or (cfg.task == "classification" and not os.path.exists(annot_csv)):
            print(f"Skipping dataset '{cfg.data.name}' (dataset files not found)")
            pytest.skip(f"Dataset '{cfg.data.name}' files not found. This is expected when testing locally.")

        print(f"Testing dataset '{cfg.data.name}' (task: {task})")

        # Function to test a dataset split
        def test_split(
            split_name,
            dataset,
            expected_image_size,
            expected_channels,
            bbox_format=None,
        ):
            if len(dataset) == 0:
                print(f"The {split_name} dataset of '{cfg.data.name}' is empty.")
                pytest.fail(f"The {split_name} dataset of '{cfg.data.name}' is empty.")

            # Test a few samples
            sample_indices = [0, len(dataset) // 2, len(dataset) - 1]
            for idx in sample_indices:
                sample = dataset[idx]
                if cfg.task == "classification":
                    image, label = sample
                    # Check label
                    assert isinstance(
                        label, int
                    ), f"Dataset '{cfg.data.name}' {split_name} split: Label is not an integer."
                    num_classes = cfg.data.num_classes
                    assert (
                        0 <= label < num_classes
                    ), f"Dataset '{cfg.data.name}' {split_name} split: Label {label} out of range [0, {num_classes - 1}]."  # noqa: E501
                elif cfg.task == "segmentation":
                    image, mask = sample
                    # Check mask shape - should be [1, H, W] for segmentation
                    assert len(mask.shape) == 3, (
                        f"Dataset '{cfg.data.name}' {split_name} split: Mask should be 3D tensor [C, H, W], "
                        f"got shape {mask.shape}"
                    )
                    assert mask.shape[0] == 1, (
                        f"Dataset '{cfg.data.name}' {split_name} split: Mask should have 1 channel, "
                        f"got {mask.shape[0]} channels"
                    )
                    assert mask.shape[1:] == image.shape[1:], (
                        f"Dataset '{cfg.data.name}' {split_name} split: Mask spatial dimensions {mask.shape[1:]} "
                        f"do not match image dimensions {image.shape[1:]}"
                    )
                    # For segmentation, check that mask values are valid
                    assert mask.min() >= 0 and mask.max() <= 1, (
                        f"Dataset '{cfg.data.name}' {split_name} split: Mask values should be between 0 and 1, "
                        f"got min={mask.min()}, max={mask.max()}"
                    )
                elif cfg.task == "detection":
                    image, target = sample
                    # Validate target format
                    assert isinstance(target, dict), (
                        f"Dataset '{cfg.data.name}' {split_name} split: "
                        f"Targets should be a dictionary, got {type(target)}"
                    )

                    # Check bounding boxes
                    bbox_key = "boxes" if "boxes" in target else "bbox"
                    class_key = "labels" if "labels" in target else "cls"
                    assert (
                        bbox_key in target
                    ), f"Dataset '{cfg.data.name}' {split_name} split: Bounding boxes missing from target dictionary"
                    assert (
                        class_key in target
                    ), f"Dataset '{cfg.data.name}' {split_name} split: Labels missing from target dictionary"

                    bboxes = target[bbox_key]
                    labels = target[class_key]
                    # Ensure bounding boxes are present
                    assert isinstance(bboxes, torch.Tensor), (
                        f"Dataset '{cfg.data.name}' {split_name} split: Expected bounding boxes to be a tensor, "
                        f"got {type(bboxes)}"
                    )
                    assert bboxes.ndim == 2 and bboxes.shape[1] == 4, (
                        f"Dataset '{cfg.data.name}' {split_name} split: Bounding boxes should have shape (N, 4), "
                        f"got {bboxes.shape}"
                    )

                    # Ensure labels are present
                    assert isinstance(labels, torch.Tensor), (
                        f"Dataset '{cfg.data.name}' {split_name} split: Expected labels to be a tensor, "
                        f"got {type(labels)}"
                    )
                    assert labels.ndim == 1, (
                        f"Dataset '{cfg.data.name}' {split_name} split: "
                        f"Labels should be a 1D tensor, got shape {labels.shape}"
                    )

                    # Ensure the number of bounding boxes matches the number of labels
                    assert bboxes.shape[0] == labels.shape[0], (
                        f"Dataset '{cfg.data.name}' {split_name} split: "
                        f"Number of bounding boxes {bboxes.shape[0]} does not match number of labels {labels.shape[0]}"
                    )

                    # Check bounding boxes based on format
                    if bbox_format == "yolo":
                        check_bboxes_yolo(bboxes, expected_image_size, split_name, cfg.data.name)
                    elif bbox_format == "coco":
                        check_bboxes_coco(
                            bboxes,
                            expected_image_size,
                            split_name,
                            cfg.data.name,
                            target,
                        )
                    elif bbox_format == "pascal_voc":
                        check_bboxes_pascal_voc(bboxes, expected_image_size, split_name, cfg.data.name)
                    else:
                        raise ValueError(f"Unsupported bbox format: {bbox_format}")

                # Check image shape
                assert image.shape[0] == expected_channels, (
                    f"Dataset '{cfg.data.name}' {split_name} split: Expected {expected_channels} channels, "
                    f"got {image.shape[0]}"
                )
                assert image.shape[1] == expected_image_size[0], (
                    f"Dataset '{cfg.data.name}' {split_name} split: Expected height {expected_image_size[0]}, "
                    f"got {image.shape[1]}"
                )
                assert image.shape[2] == expected_image_size[1], (
                    f"Dataset '{cfg.data.name}' {split_name} split: Expected width {expected_image_size[1]}, "
                    f"got {image.shape[2]}"
                )

        if cfg.task == "detection":
            bbox_formats = ["yolo", "coco", "pascal_voc"]
            for bbox_format in bbox_formats:
                cfg.model.bbox_format = bbox_format

                # Get transforms
                transforms = get_transforms(cfg)

                # Expected image size and channels
                expected_image_size = cfg.transforms.image_size
                expected_channels = 1 if cfg.data.image_type == "grayscale" else 3

                # Get datasets
                train_dataset, val_dataset, test_dataset = initialize_datasets(cfg, transforms, bbox_format)

                test_split(
                    "train",
                    train_dataset,
                    expected_image_size,
                    expected_channels,
                    bbox_format,
                )
                test_split(
                    "val",
                    val_dataset,
                    expected_image_size,
                    expected_channels,
                    bbox_format,
                )
                test_split(
                    "test",
                    test_dataset,
                    expected_image_size,
                    expected_channels,
                    bbox_format,
                )

        else:
            # Get transforms
            transforms = get_transforms(cfg)

            # Get datasets
            train_dataset, val_dataset, test_dataset = initialize_datasets(cfg, transforms)

            # Expected image size and channels
            expected_image_size = cfg.transforms.image_size
            expected_channels = 1 if cfg.data.image_type == "grayscale" else 3

            # Test each split
            test_split("train", train_dataset, expected_image_size, expected_channels)
            test_split("validation", val_dataset, expected_image_size, expected_channels)
            test_split("test", test_dataset, expected_image_size, expected_channels)
