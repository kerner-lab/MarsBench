"""
"""


import os
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union

import pandas as pd

from .BaseClassificationDataset import BaseClassificationDataset


class Atmospheric_Dust_Classification_EDR(BaseClassificationDataset):
    """
    """

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
        data_dir = data_dir + f"/{split}"
        super(Atmospheric_Dust_Classification_EDR, self).__init__(cfg, data_dir, transform)

    def _load_data(self) -> Tuple[List[str], List[str], List[int]]:
        image_ids = self.annot["file_id"].astype(str).tolist()
        feature_names = self.annot["feature_name"].astype(str).tolist()
        labels = self.annot["label"].astype(int).tolist()
        return image_ids, feature_names, labels
