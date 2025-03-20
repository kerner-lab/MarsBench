"""Utility functions for testing detection datasets."""


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
        f"Dataset '{dataset_name}' {split_name} split (bbox_format: pascal_voc): " f"xmax should be greater than xmin."
    )
    assert (ymax > ymin).all(), (
        f"Dataset '{dataset_name}' {split_name} split (bbox_format: " f"pascal_voc): ymax should be greater than ymin."
    )
    assert (xmin >= 0).all() & (xmax <= expected_image_size[1]).all(), (
        f"Dataset '{dataset_name}' {split_name} split (bbox_format: pascal_voc): "
        f"xmin and xmax should be within image width range [0, {expected_image_size[1]}]."
    )
    assert (ymin >= 0).all() & (ymax <= expected_image_size[0]).all(), (
        f"Dataset '{dataset_name}' {split_name} split (bbox_format: pascal_voc): "
        f"ymin and ymax should be within image height range [0, {expected_image_size[0]}]."
    )
