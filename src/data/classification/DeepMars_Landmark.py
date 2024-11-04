import os
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
import torch

from .BaseClassificationDataset import BaseClassificationDataset


class DeepMars_Landmark(BaseClassificationDataset):
    """
    DeepMars_Landmark https://zenodo.org/records/1048301
    """

    def __init__(
        self,
        cfg,
        data_dir,
        transform,
        annot_csv: Union[str, os.PathLike],
        split: Literal["train", "val", "test"] = "train",
        generator: Optional[torch.Generator] = None,
    ):
        self.cfg = cfg
        self.annot = pd.read_csv(annot_csv)
        generator = (
            torch.Generator().manual_seed(cfg.seed) if generator is None else generator
        )
        total_size = len(self.annot)
        self.indices = self.determine_data_splits(total_size, generator, split)
        super(DeepMars_Landmark, self).__init__(cfg, data_dir, transform)

    def _load_data(self) -> Tuple[List[str], List[int]]:
        annot_subset = (
            self.annot if self.indices is None else self.annot.iloc[self.indices]
        )
        image_paths = annot_subset["image_path"].astype(str).tolist()
        labels = annot_subset["label"].astype(int).tolist()
        return image_paths, labels
