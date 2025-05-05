"""
"""

import os
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union

import pandas as pd

from .BaseClassificationDataset import BaseClassificationDataset


class Atmospheric_Dust_Classification_RDR(BaseClassificationDataset):
    """ """

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
        super(Atmospheric_Dust_Classification_RDR, self).__init__(cfg, data_dir, transform)

    def _load_data(self) -> Tuple[List[str], List[int]]:
        image_ids = self.annot["file_id"].astype(str).tolist()
        feature_names = self.annot["feature_name"].astype(str).tolist()
        gts = self.annot["label"].astype(int).tolist()
        image_paths = [
            os.path.join(self.data_dir, "data", self.split, feature_name, f"{image_id}")
            for image_id, feature_name in zip(image_ids, feature_names)
        ]
        return image_paths, gts
