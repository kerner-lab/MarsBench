"""
ConeQuest dataset for Mars volcanic cone detection.
"""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

import pandas as pd
from lxml import etree
from omegaconf import DictConfig

from .BaseDetectionDataset import BaseDetectionDataset

logger = logging.getLogger(__name__)


class ConeQuest_Detection(BaseDetectionDataset):
    def __init__(
        self,
        cfg: DictConfig,
        data_dir: str,
        transform=None,
        bbox_format: Literal["coco", "yolo", "pascal_voc"] = "yolo",
        annot_csv: Optional[str] = None,
        split: Literal["train", "val", "test"] = "train",
    ):
        self.annot_csv = annot_csv
        super().__init__(cfg, data_dir, transform, bbox_format, split)

    def _load_data(self) -> Tuple[List[str], List[List[float]], List[List[int]], List[str]]:
        image_paths = sorted(os.listdir(Path(self.data_dir) / "data" / self.split / "images"))
        image_suffix = Path(image_paths[0]).suffix
        names = [Path(p).stem for p in image_paths]

        if self.bbox_format == "yolo":
            bbox_paths = sorted(os.listdir(Path(self.data_dir) / "data" / self.split / "labels"))
            bbox_paths = [os.path.join(self.data_dir, "data", self.split, "labels", p) for p in bbox_paths]

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
            coco_json_path = os.path.join(self.data_dir, "data", self.split, "coco_annotations.json")

            with open(coco_json_path, "r") as file:
                coco_annotations = json.load(file)
            img_id_to_filename = {img["id"]: Path(img["file_name"]).stem for img in coco_annotations["images"]}

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
            bbox_paths = sorted(os.listdir(Path(self.data_dir) / "data" / self.split / "pascal_voc"))
            bbox_paths = [os.path.join(self.data_dir, "data", self.split, "pascal_voc", p) for p in bbox_paths]

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

        image_paths = [os.path.join(self.data_dir, "data", self.split, "images", p + image_suffix) for p in valid_names]

        logger.info(f"Loaded {len(image_paths)} images with annotations")
        logger.info(f"Using annotations from {self.annot_csv}")
        if self.annot_csv is not None:
            self.annot = pd.read_csv(self.annot_csv)
            selected_names = set(self.annot[self.annot["split"] == self.split]["file_id"].tolist())
            selected_stems = {Path(fid).stem for fid in selected_names}
            valid_names = [name for name in valid_names if name in selected_stems]
            image_paths = [
                os.path.join(self.data_dir, "data", self.split, "images", name + image_suffix) for name in valid_names
            ]
            annotation_list = [annotations[name] for name in valid_names]
            label_list = [labels[name] for name in valid_names]
            return image_paths, annotation_list, label_list, valid_names

        return (
            image_paths,
            list(annotations.values()),
            list(labels.values()),
            valid_names,
        )
