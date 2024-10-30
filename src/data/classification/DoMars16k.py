import glob
import os
from itertools import chain
from typing import List
from typing import Tuple

from .BaseClassificationDataset import BaseClassificationDataset


class DoMars16k(BaseClassificationDataset):
    """
    DoMars16k dataset https://zenodo.org/records/4291940
    """

    def __init__(self, cfg, data_dir, transform):
        super(DoMars16k, self).__init__(cfg, data_dir, transform)

    def _load_data(self) -> Tuple[List[str], List[int]]:
        image_paths = []
        labels = []
        extensions = self.cfg.data.valid_image_extensions
        for label, class_dir in enumerate(os.listdir(self.data_dir)):
            class_dir_path = os.path.join(self.data_dir, class_dir)
            matched_files = list(
                chain.from_iterable(
                    glob.glob(os.path.join(class_dir_path, f"*.{ext}"))
                    for ext in extensions
                )
            )
            image_paths.extend(matched_files)
            labels.extend([label] * len(matched_files))
        return image_paths, labels
