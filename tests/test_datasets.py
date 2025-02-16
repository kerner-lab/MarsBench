import glob
import os

import pytest
import torch
from hydra import compose
from hydra import initialize_config_dir

from src.data import get_dataset
from src.utils.transforms import get_transforms


def initialize_datasets(cfg, transforms, bbox_format=None):
    if cfg.task == "classification":
        train_dataset, val_dataset, test_dataset = get_dataset(
            cfg, transforms[:2], subset=cfg.test.data.subset_size
        )
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


def check_bboxes_yolo(bboxes, expected_image_size, split_name, dataset_name):
    x_center, y_center, width, height = (
        bboxes[:, 0],
        bboxes[:, 1],
        bboxes[:, 2],
        bboxes[:, 3],
    )

    assert (x_center >= 0).all() & (
        x_center <= 1
    ).all(), f"Dataset '{dataset_name}' {split_name} split (bbox_format: yolo): x_center should be within range [0, 1]."
    assert (y_center >= 0).all() & (
        y_center <= 1
    ).all(), f"Dataset '{dataset_name}' {split_name} split (bbox_format: yolo): y_center should be within range [0, 1]."
    assert (width >= 0).all() & (
        width <= 1
    ).all(), f"Dataset '{dataset_name}' {split_name} split (bbox_format: yolo): width should be within range [0, 1]."
    assert (height >= 0).all() & (
        height <= 1
    ).all(), f"Dataset '{dataset_name}' {split_name} split (bbox_format: yolo): height should be within range [0, 1]."


def check_bboxes_coco(bboxes, expected_image_size, split_name, dataset_name, target):
    x_min, y_min, width, height = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    assert (x_min + width <= expected_image_size[1]).all(), (
        f"Dataset '{dataset_name}' {split_name} split (bbox_format: coco): "
        f"x_min + width should be less than or equal to {expected_image_size[1]}."
    )
    assert (x_min >= 0).all() & (x_min + width <= expected_image_size[1]).all(), (
        f"Dataset '{dataset_name}' {split_name} split (bbox_format: coco): "
        f"x_min and width should be within image width range [0, {expected_image_size[1]}]."
    )
    assert (y_min >= 0).all() & (y_min + height <= expected_image_size[0]).all(), (
        f"Dataset '{dataset_name}' {split_name} split (bbox_format: coco): "
        f"y_min and height should be within image height range [0, {expected_image_size[0]}]."
    )


def check_bboxes_pascal_voc(bboxes, expected_image_size, split_name, dataset_name):
    xmin, ymin, xmax, ymax = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    assert (xmax > xmin).all(), (
        f"Dataset '{dataset_name}' {split_name} split (bbox_format: pascal_voc): "
        f"xmax should be greater than xmin."
    )
    assert (ymax > ymin).all(), (
        f"Dataset '{dataset_name}' {split_name} split (bbox_format: "
        f"pascal_voc): ymax should be greater than ymin."
    )
    assert (xmin >= 0).all() & (xmax <= expected_image_size[1]).all(), (
        f"Dataset '{dataset_name}' {split_name} split (bbox_format: pascal_voc): "
        f"xmin and xmax should be within image width range [0, {expected_image_size[1]}]."
    )
    assert (ymin >= 0).all() & (ymax <= expected_image_size[0]).all(), (
        f"Dataset '{dataset_name}' {split_name} split (bbox_format: pascal_voc): "
        f"ymin and ymax should be within image height range [0, {expected_image_size[0]}]."
    )


@pytest.mark.parametrize("dataset_config_file", glob.glob("configs/data/*.yaml"))
def test_datasets(dataset_config_file):
    # Get dataset name from config file name
    dataset_name = os.path.splitext(os.path.basename(dataset_config_file))[0]
    config_dir = os.path.abspath("configs")

    # Initialize Hydra and compose configuration
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name="config", overrides=[f"data={dataset_name}"])

        # Check dataset status
        if cfg.data.status not in cfg.test.data.status:
            print(f"Skipping dataset '{dataset_name}' (status: {cfg.data.status})")
            pytest.skip(
                f"Dataset '{dataset_name}' is not in status set {cfg.test.data.status} for testing."
            )

        print(f"Testing dataset '{dataset_name}'")

        # Get transforms
        transforms = get_transforms(cfg)

        # Expected image size and channels
        expected_image_size = cfg.transforms.image_size
        expected_channels = 1 if cfg.data.image_type == "grayscale" else 3

        # Function to test a dataset split
        def test_split(split_name, dataset, bbox_format=None):
            if len(dataset) == 0:
                print(f"The {split_name} dataset of '{dataset_name}' is empty.")
                pytest.fail(f"The {split_name} dataset of '{dataset_name}' is empty.")

            # Test a few samples
            sample_indices = [0, len(dataset) // 2, len(dataset) - 1]
            for idx in sample_indices:
                sample = dataset[idx]
                if cfg.task == "classification":
                    image, label = sample
                    # Check label
                    assert isinstance(
                        label, int
                    ), f"Dataset '{dataset_name}' {split_name} split: Label is not an integer."
                    num_classes = cfg.data.num_classes
                    assert (
                        0 <= label < num_classes
                    ), f"Dataset '{dataset_name}' {split_name} split: Label {label} out of range [0, {num_classes - 1}]."  # noqa: E501
                elif cfg.task == "segmentation":
                    image, mask = sample
                    # Check mask shape
                    assert mask.shape == image.shape, (
                        f"Dataset '{dataset_name}' {split_name} split: Mask shape {mask.shape} "
                        f"does not match image shape {image.shape}"
                    )
                    # For segmentation, check that mask values are valid
                    assert mask.min() >= 0 and mask.max() <= 1, (
                        f"Dataset '{dataset_name}' {split_name} split: Mask values should be between 0 and 1, "
                        f"got min={mask.min()}, max={mask.max()}"
                    )
                elif cfg.task == "detection":
                    image, target = sample
                    # Validate target format
                    assert isinstance(target, dict), (
                        f"Dataset '{dataset_name}' {split_name} split: "
                        f"Targets should be a dictionary, got {type(target)}"
                    )

                    # Check bounding boxes
                    bbox_key = "boxes" if "boxes" in target else "bbox"
                    class_key = "labels" if "labels" in target else "cls"
                    assert (
                        bbox_key in target
                    ), f"Dataset '{dataset_name}' {split_name} split: Bounding boxes missing from target dictionary"
                    assert (
                        class_key in target
                    ), f"Dataset '{dataset_name}' {split_name} split: Labels missing from target dictionary"

                    bboxes = target[bbox_key]
                    labels = target[class_key]
                    # Ensure bounding boxes are present
                    assert isinstance(bboxes, torch.Tensor), (
                        f"Dataset '{dataset_name}' {split_name} split: Expected bounding boxes to be a tensor, "
                        f"got {type(bboxes)}"
                    )
                    assert bboxes.ndim == 2 and bboxes.shape[1] == 4, (
                        f"Dataset '{dataset_name}' {split_name} split: Bounding boxes should have shape (N, 4), "
                        f"got {bboxes.shape}"
                    )

                    # Ensure labels are present
                    assert isinstance(labels, torch.Tensor), (
                        f"Dataset '{dataset_name}' {split_name} split: Expected labels to be a tensor, "
                        f"got {type(labels)}"
                    )
                    assert labels.ndim == 1, (
                        f"Dataset '{dataset_name}' {split_name} split: "
                        f"Labels should be a 1D tensor, got shape {labels.shape}"
                    )

                    # Ensure the number of bounding boxes matches the number of labels
                    assert bboxes.shape[0] == labels.shape[0], (
                        f"Dataset '{dataset_name}' {split_name} split: "
                        f"Number of bounding boxes {bboxes.shape[0]} does not match number of labels {labels.shape[0]}"
                    )

                    # Check bounding boxes based on format
                    if bbox_format == "yolo":
                        check_bboxes_yolo(
                            bboxes, expected_image_size, split_name, dataset_name
                        )
                    elif bbox_format == "coco":
                        check_bboxes_coco(
                            bboxes,
                            expected_image_size,
                            split_name,
                            dataset_name,
                            target,
                        )
                    elif bbox_format == "pascal_voc":
                        check_bboxes_pascal_voc(
                            bboxes, expected_image_size, split_name, dataset_name
                        )
                    else:
                        raise ValueError(f"Unsupported bbox format: {bbox_format}")

                # Check image shape
                assert image.shape[0] == expected_channels, (
                    f"Dataset '{dataset_name}' {split_name} split: Expected {expected_channels} channels, "
                    f"got {image.shape[0]}"
                )
                assert image.shape[1] == expected_image_size[0], (
                    f"Dataset '{dataset_name}' {split_name} split: Expected height {expected_image_size[0]}, "
                    f"got {image.shape[1]}"
                )
                assert image.shape[2] == expected_image_size[1], (
                    f"Dataset '{dataset_name}' {split_name} split: Expected width {expected_image_size[1]}, "
                    f"got {image.shape[2]}"
                )

                print(
                    f"Dataset '{dataset_name}' {split_name} split, index {idx}: "
                    f"Image shape {image.shape}"
                    + (
                        f", Mask shape {mask.shape}"
                        if cfg.task == "segmentation"
                        else ""
                    )
                    + (
                        f", BBox shape {bboxes.shape}, Labels shape {labels.shape}"
                        if cfg.task == "detection"
                        else ""
                    )
                )

        # Test each split
        if cfg.task == "detection":
            bbox_formats = ["yolo", "coco", "pascal_voc"]
            for bbox_format in bbox_formats:
                # Get datasets
                train_dataset, val_dataset, test_dataset = initialize_datasets(
                    cfg, transforms, bbox_format
                )
                test_split("train", train_dataset, bbox_format)
                test_split("val", val_dataset, bbox_format)
                test_split("test", test_dataset, bbox_format)
        else:
            # Get datasets
            train_dataset, val_dataset, test_dataset = initialize_datasets(
                cfg, transforms
            )
            test_split("train", train_dataset)
            test_split("val", val_dataset)
            test_split("test", test_dataset)
