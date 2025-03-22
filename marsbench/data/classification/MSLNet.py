"""
Mars Science Laboratory (MSL) image dataset for Mars rover image classification.
"""

import os
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union

import pandas as pd

from .BaseClassificationDataset import BaseClassificationDataset


class MSLNet(BaseClassificationDataset):
    """
    Mars Image Content Classification Mastcam & MAHILI Dataset
    https://zenodo.org/records/4033453
    """

    def __init__(
        self,
        cfg,
        data_dir,
        transform,
        annot_csv: Union[str, os.PathLike],
        split: Literal["train", "val", "test"] = "train",
    ):
        self.annot = pd.read_csv(annot_csv)
        self.annot = self.annot[self.annot["split"] == split]
        super(MSLNet, self).__init__(cfg, data_dir, transform)

    def _load_data(self) -> Tuple[List[str], List[int]]:
        image_paths = self.annot["image_path"].astype(str).tolist()
        labels = self.annot["label"].astype(int).tolist()
        return image_paths, labels
