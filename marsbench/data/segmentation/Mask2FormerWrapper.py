"""
Wrapper class to adapt any segmentation dataset for Mask2Former.
"""

import cv2
from torch.utils.data import Dataset


class Mask2FormerWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        image = cv2.imread(self.dataset.image_paths[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("uint8")
        mask = cv2.imread(self.dataset.ground[index], 0).astype("float32")

        transformed = self.dataset.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        orig_image = image.clone()
        orig_mask = mask.clone()

        return image, mask, orig_image, orig_mask
