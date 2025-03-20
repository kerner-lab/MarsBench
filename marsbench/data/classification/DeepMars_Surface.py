import os
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union

import pandas as pd

from .BaseClassificationDataset import BaseClassificationDataset


class DeepMars_Surface(BaseClassificationDataset):
    """
    DeepMars_Surface https://zenodo.org/records/1049137
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
        self.split = split
        super(DeepMars_Surface, self).__init__(cfg, data_dir, transform)

    def _load_data(self) -> Tuple[List[str], List[int]]:
        annot_subset = self.annot[self.annot["split"] == self.split]
        image_paths = annot_subset["image_path"].astype(str).tolist()
        labels = annot_subset["label"].astype(int).tolist()
        return image_paths, labels
