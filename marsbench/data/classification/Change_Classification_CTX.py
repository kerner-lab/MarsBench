"""
Planetary Surface Features Change Detection Dataset
"""

import os
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .BaseClassificationDataset import BaseClassificationDataset


class Change_Classification_CTX(BaseClassificationDataset):
    def __init__(
        self,
        cfg,
        data_dir,
        transform,
        annot_csv: Union[str, os.PathLike],
        split: Literal["train", "val", "test"] = "train",
    ):
        self.split = split
        self.annot = pd.read_csv(annot_csv)
        self.annot = self.annot[self.annot["split"] == split]
        # data_dir = data_dir + f"/{split}"
        super(Change_Classification_CTX, self).__init__(cfg, data_dir, transform)

    def _load_data(self) -> Tuple[List[str], List[int]]:
        image_ids = self.annot["file_id"].astype(str).tolist()
        feature_names = self.annot["feature_name"].astype(str).tolist()
        labels = self.annot["label"].astype(int).tolist()
        image_paths = [
            os.path.join(self.data_dir, "data", self.split, feature_name, f"{image_id}")
            for image_id, feature_name in zip(image_ids, feature_names)
        ]
        return image_paths, labels

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_id_before = self.image_paths[idx]
        image_id_after = self.image_paths[idx].replace("before", "after")

        layer_before = np.array(Image.open(image_id_before))
        layer_zero = np.zeros((layer_before.shape[0], layer_before.shape[1]), dtype=layer_before.dtype)
        layer_after = np.array(Image.open(image_id_after))

        image = np.stack([layer_zero, layer_after, layer_before], axis=-1)
        label = self.labels[idx]

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image)

        return image, label
