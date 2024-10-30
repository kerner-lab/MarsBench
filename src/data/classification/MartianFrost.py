from pathlib import Path
from typing import List
from typing import Tuple

from .BaseClassificationDataset import BaseClassificationDataset


class MartianFrost(BaseClassificationDataset):
    """
    Martian Frost dataset
    https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=doi:10.48577/jpl.QJ9PYA
    """

    def __init__(self, cfg, data_dir, transform, txt_file):
        self.data_dir = data_dir
        self.txt_file = txt_file
        super(MartianFrost, self).__init__(cfg, data_dir, transform)

    def _load_data(self) -> Tuple[List[str], List[int]]:
        image_paths = []
        labels = []

        with open(self.txt_file, "r", encoding="utf-8") as text:
            valid_parents = set(line.strip() for line in text)

        data_dir = Path(self.data_dir)

        patterns = [("frost", 1), ("background", 0)]

        for subfolder, label in patterns:
            for image_path in data_dir.glob(f"*/tiles/{subfolder}/*"):
                each_folder = image_path.parents[2].name
                parent_directory = each_folder[:15]
                if parent_directory in valid_parents:
                    image_paths.append(str(image_path))
                    labels.append(label)

        return image_paths, labels
