import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from lxml import etree
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDetectionDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        data_dir: str,
        transform=None,
        bbox_format: Literal["coco", "yolo", "pascal_voc"] = None,
        split: Literal["train", "val", "test"] = "train",
    ):
        self.cfg = cfg
        IMAGE_MODES = {"rgb": "RGB", "grayscale": "L", "l": "L"}
        requested_mode = cfg.data.image_type.lower().strip()
        self.image_type = IMAGE_MODES.get(requested_mode)
        if self.image_type is None:
            logger.error(
                f"Invalid/unsupported image_type '{requested_mode}'. Valid options are: {list(IMAGE_MODES.keys())}. "
                "Defaulting to RGB."
            )
            self.image_type = "RGB"
        self.data_dir = data_dir
        self.transform = transform
        self.bbox_format = bbox_format
        self.split = split

        logger.info(
            f"Loading {self.__class__.__name__} from {data_dir} (split: {split})"
        )
        (
            self.image_paths,
            self.annotations,
            self.labels,
            _,  # image_ids
        ) = self._load_data()
        logger.info(f"Loaded {len(self.image_paths)} images with annotations")

        # Validate image extensions
        for image_path in self.image_paths:
            if not image_path.endswith(tuple(cfg.data.valid_image_extensions)):
                logger.error(f"Invalid image format: {image_path}")
                raise ValueError(f"Invalid image format: {image_path}")

        logger.info(
            f"Dataset initialized with mode: {self.image_type}, "
            f"transforms: {'applied' if transform else 'none'}, "
        )

    def _load_data(self):
        image_paths = sorted(os.listdir(Path(self.data_dir) / self.split / "images"))
        image_suffix = Path(image_paths[0]).suffix
        names = [Path(p).stem for p in image_paths]

        if self.bbox_format == "yolo":
            bbox_paths = sorted(os.listdir(Path(self.data_dir) / self.split / "labels"))
            bbox_paths = [
                os.path.join(self.data_dir, self.split, "labels", p) for p in bbox_paths
            ]

            annotations = defaultdict(list)
            labels = defaultdict(list)
            for bbox_path in bbox_paths:
                annotation_key = Path(bbox_path).stem
                with open(bbox_path, "r") as file:
                    for line in file:
                        annots = list(map(float, line.strip().split()))
                        class_id = int(annots[0])
                        bbox = annots[1:5]
                        annotations[annotation_key].append(bbox)
                        labels[annotation_key].append(class_id)
            annotations = dict(annotations)
            labels = dict(labels)

        elif self.bbox_format == "coco":
            coco_json_path = os.path.join(
                self.data_dir, self.split, "coco_annotations.json"
            )

            with open(coco_json_path, "r") as file:
                coco_annotations = json.load(file)
            img_id_to_filename = {
                img["id"]: Path(img["file_name"]).stem
                for img in coco_annotations["images"]
            }

            annotations = defaultdict(list)
            labels = defaultdict(list)
            for annotation in coco_annotations["annotations"]:
                image_id = annotation["image_id"]
                bbox = annotation["bbox"]
                file_name = img_id_to_filename[image_id]
                annotations[file_name].append(bbox)
                labels[file_name].append(annotation["category_id"])
            annotations = dict(sorted(annotations.items()))
            labels = dict(sorted(labels.items()))

        elif self.bbox_format == "pascal_voc":
            bbox_paths = sorted(
                os.listdir(Path(self.data_dir) / self.split / "pascal_voc")
            )
            bbox_paths = [
                os.path.join(self.data_dir, self.split, "pascal_voc", p)
                for p in bbox_paths
            ]

            annotations = defaultdict(list)
            labels = defaultdict(list)
            class_to_label = {}
            for bbox_path in bbox_paths:
                tree = etree.parse(bbox_path)
                root = tree.getroot()

                for obj in root.findall("object"):
                    bbox = obj.find("bndbox")
                    annotation_key = Path(bbox_path).stem
                    bbox = [
                        int(bbox.find("xmin").text),
                        int(bbox.find("ymin").text),
                        int(bbox.find("xmax").text),
                        int(bbox.find("ymax").text),
                    ]
                    annotations[annotation_key].append(bbox)

                    class_name = obj.find("name").text
                    if class_name not in class_to_label:
                        class_to_label[class_name] = len(class_to_label) + 1
                    label = class_to_label[class_name]
                    labels[annotation_key].append(label)
            annotations = dict(annotations)
            labels = dict(labels)

        names_with_bbox = list(annotations.keys())
        valid_names = [name for name in names if name in names_with_bbox]

        if valid_names == names_with_bbox and valid_names == list(labels.keys()):
            logging.info("name and annotations in sync.")
        else:
            logging.warning("names and annotations are not in sync.")

        image_paths = [
            os.path.join(self.data_dir, self.split, "images", p + image_suffix)
            for p in valid_names
        ]

        return (
            image_paths,
            list(annotations.values()),
            list(labels.values()),
            valid_names,
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert(self.image_type)
        bboxes = self.annotations[idx]
        labels = self.labels[idx]

        img_width, img_height = image.size
        image = np.array(image)

        if self.transform:
            transformed = self.transform(
                image=image, bboxes=bboxes, class_labels=labels
            )

        image = transformed["image"]
        img_height, img_width = image.shape[-2:]
        bboxes = transformed["bboxes"]
        labels = transformed["class_labels"]
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.cfg.model.name.lower() == "efficientdet":
            bboxes = bboxes[:, [1, 0, 3, 2]]
            bbox_label = "bbox"
            class_label = "cls"
        else:
            bbox_label = "boxes"
            class_label = "labels"

        target = {
            bbox_label: bboxes,
            class_label: labels,
            "image_id": torch.tensor([idx]),
            "img_size": torch.tensor([img_height, img_width]),
            "img_scale": torch.tensor([1.0]),
        }

        return image, target
