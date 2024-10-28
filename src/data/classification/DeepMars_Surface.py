from typing import List, Tuple
from .BaseClassificationDataset import BaseClassificationDataset
import os


class DeepMars_Surface(BaseClassificationDataset):
    """
    DeepMars_Surface https://zenodo.org/records/1049137
    """
    def __init__(self, cfg, data_dir, transform, txt_file):
        self.txt_file = txt_file
        super(DeepMars_Surface, self).__init__(cfg, data_dir, transform)

    def _load_data(self) -> Tuple[List[str], List[int]]:
        image_paths = []
        labels = []

        with open(self.txt_file, "r", encoding="utf-8") as text:
            for line in text:
                image_name, label_str = line.strip().split()[:2]
                label = int(label_str)
                if label == 23:
                    label = 22
                elif label == 24:
                    label = 23
                image_paths.append(os.path.join(self.data_dir, image_name))
                labels.append(label)

        return image_paths, labels

