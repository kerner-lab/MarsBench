from typing import List, Optional, Tuple, Sequence
from .BaseClassificationDataset import BaseClassificationDataset
import os


class DeepMars_Landmark(BaseClassificationDataset):
    """
    DeepMars_Landmark https://zenodo.org/records/1048301
    """

    def __init__(self, cfg, data_dir, transform, txt_file, indices: Optional[Sequence[int]] = None):
        self.txt_file = txt_file
        self.indices = indices
        super(DeepMars_Landmark, self).__init__(cfg, data_dir, transform)

    def _load_data(self) -> Tuple[List[str], List[int]]:
        image_paths = []
        labels = []

        with open(self.txt_file, "r", encoding="utf-8") as text:
            for line in text:
                image_name, label_str = line.strip().split()[:2]
                label = int(label_str)
                image_paths.append(os.path.join(self.data_dir, image_name))
                labels.append(5 if label == 6 else label)

        if self.indices is not None:
            image_paths = [image_paths[i] for i in self.indices]
            labels = [labels[i] for i in self.indices]

        return image_paths, labels

