import os
from typing import List
from typing import Tuple

from .BaseClassificationDataset import BaseClassificationDataset


class MSLNet(BaseClassificationDataset):
    """
    Mars Image Content Classification Mastcam & MAHILI Dataset
    https://zenodo.org/records/4033453
    """

    def __init__(self, cfg, data_dir, transform, txt_file):
        self.txt_file = txt_file
        super(MSLNet, self).__init__(cfg, data_dir, transform)

    def _load_data(self) -> Tuple[List[str], List[int]]:
        image_paths = []
        labels = []

        with open(self.txt_file, "r", encoding="utf-8") as text:
            for line in text:
                image_name, label_str = line.strip().split()[:2]
                image_paths.append(os.path.join(self.data_dir, image_name))
                labels.append(int(label_str))

        return image_paths, labels
