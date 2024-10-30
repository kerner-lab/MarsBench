import os
from typing import List
from typing import Tuple

from .BaseClassificationDataset import BaseClassificationDataset


class HiRISENet(BaseClassificationDataset):
    """
    Mars Image Content Classfication-HiRISENet https://zenodo.org/records/4002935
    """

    def __init__(self, cfg, data_dir, transform, txt_file, split_type):
        self.txt_file = txt_file
        self.split_type = split_type
        super(HiRISENet, self).__init__(cfg, data_dir, transform)

    def _load_data(self) -> Tuple[List[str], List[int]]:
        image_paths = []
        labels = []

        with open(self.txt_file, "r", encoding="utf-8") as text:
            for line in text:
                image_name, class_type_str, split_style = line.strip().split()[:3]
                if self.split_type == split_style:
                    image_paths.append(os.path.join(self.data_dir, image_name))
                    labels.append(int(class_type_str))

        return image_paths, labels
