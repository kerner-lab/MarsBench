"""
DoMars16k dataset for Mars surface feature classification.
"""

import os
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union

import pandas as pd

from .BaseClassificationDataset import BaseClassificationDataset


class DoMars16k(BaseClassificationDataset):
    """
    DoMars16k dataset https://zenodo.org/records/4291940
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
        super(DoMars16k, self).__init__(cfg, data_dir, transform)

    def _load_data(self) -> Tuple[List[str], List[int]]:
        image_paths = self.annot["image_path"].astype(str).tolist()
        labels = self.annot["label"].astype(int).tolist()
        return image_paths, labels
