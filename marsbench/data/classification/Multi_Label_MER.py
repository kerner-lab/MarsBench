"""
MER Opportunity and Spirit Rovers Pancam Images Labeled Data Set
"""

import os
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union

import pandas as pd

from .BaseClassificationDataset import BaseClassificationDataset


class Multi_Label_MER(BaseClassificationDataset):
    """
    MER Opportunity and Spirit Rovers Pancam Images Labeled Data Set
    https://zenodo.org/records/4302760
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
        super(Multi_Label_MER, self).__init__(cfg, data_dir, transform)

    def _load_data(self) -> Tuple[List[str], List[int]]:
        image_ids = self.annot["file_id"].astype(str).tolist()
        labels = self.annot["label"].tolist()
        image_paths = [os.path.join(self.data_dir, "data", self.split, f"{image_id}") for image_id in image_ids]
        return image_paths, labels
