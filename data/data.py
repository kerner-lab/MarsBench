
import os
from torch.utils.data import Dataset
from skimage import io
import numpy as np
from PIL import Image
import cv2
import json
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, data_dir=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_path, self.labels = self._getdata()

    def _getdata(self):
       raise NotImplementedError('Not Implemented in subclass')

    def __len__(self):
       return len(self.labels)

    def __getitem__(self, ind):
        image = cv2.imread(self.image_path[ind], cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        label = self.labels[ind]

        if self.transform:
            image = self.transform(image)

        return image,label
    


# Mars Image Content Classification-Landmark Dataset subclass
class MarsDatasetLandmark(CustomDataset): 
    def __init__(self, data_dir, transform, txt_file, split_type):
        self.text_file = txt_file
        self.split_type = split_type
        super(MarsDatasetLandmark, self).__init__(data_dir, transform)
       

    def _getdata(self):
        image_path = []
        labels = []
        with open(self.text_file) as text:
            lines= list(text)
            image_names = [block.split()[0] for block in lines]
            class_types = [block.split()[1] for block in lines]
            split_styles = [block.split()[2] for block in lines]
            for image_name, class_type, split_style in zip(image_names,class_types, split_styles):
                
                if self.split_type == 'train' and split_style == 'train':
                    image_path.append(os.path.join(self.data_dir, image_name))
                    labels.append(int(class_type))

                if self.split_type == 'val' and split_style == 'val':
                    image_path.append(os.path.join(self.data_dir, image_name))
                    labels.append(int(class_type))

                if self.split_type == 'test' and split_style == 'test':
                    image_path.append(os.path.join(self.data_dir, image_name))
                    labels.append(int(class_type))
        return image_path, labels 

class MarsDatasetSurface(CustomDataset):
    def __init__(self, data_dir, transform, txt_file):
        self.text_file = txt_file
        super(MarsDatasetSurface, self).__init__(data_dir, transform)
    
    def _getdata(self):
        image_path = []
        labels = []

        with open(self.text_file) as text:
            lines = list(text)
            image_names = [block.split()[0] for block in lines]
            class_types = [block.split()[1] for block in lines]

            for image_name, label in zip(image_names,class_types):
                image_path.append(os.path.join(self.data_dir, image_name))
                labels.append(int(label))
        return image_path, labels   



#DoMars16k dataset subclass
class DoMars16k(CustomDataset):
    def __init__(self, data_dir, transform):
        super(DoMars16k,self).__init__(data_dir, transform)
    
    def _getdata(self):
        image_path = []
        labels = []
        for label, class_dir in enumerate(os.listdir(self.data_dir)):
            class_dir_path = os.path.join(self.data_dir, class_dir)
            
            for image in os.listdir(class_dir_path):
                if image.endswith(('jpg','png','jpeg')):
                    image_path.append(os.path.join(class_dir_path,image))
                    labels.append(label)
        
        return image_path, labels

class DeepMars_Landmark(CustomDataset):
    def __init__(self, data_dir, transform, txt_file):
        self.text_file = txt_file
        super(DeepMars_Landmark, self).__init__(data_dir, transform)
    
    def _getdata(self):
        image_path = []
        labels = []

        with open(self.text_file) as text:
            lines = list(text)
            image_names = [block.split()[0] for block in lines]
            class_types = [block.split()[1] for block in lines]

            for image_name, label in zip(image_names,class_types):
                image_path.append(os.path.join(self.data_dir, image_name))
                if int(label)==6:
                    labels.append(5)
                else:
                   labels.append(int(label))
        
        return image_path, labels


class DeepMars_Surface(CustomDataset):
    def __init__(self, data_dir, transform, txt_file):
        self.text_file = txt_file
        super(DeepMars_Surface, self).__init__(data_dir, transform)
    
    def _getdata(self):
        image_path = []
        labels = []

        with open(self.text_file) as text:
            lines = list(text)
            image_names = [block.split()[0] for block in lines]
            class_types = [block.split()[1] for block in lines]

            for image_name, label in zip(image_names,class_types):
                image_path.append(os.path.join(self.data_dir, image_name))
                if int(label)==23:
                    labels.append(22)
                
                elif int(label)==24:
                   labels.append(int(23))
                
                else:
                    labels.append(int(label))
        return image_path, labels
    

class MartianFrost(CustomDataset):

    def __init__(self, data_dir, transform, txt_file):
        self.data_dir = data_dir
        self.text_file = txt_file
        super(MartianFrost, self).__init__(data_dir, transform)

    def _getdata(self):
        image_path, labels = [], []

        with open(self.text_file) as text:
            lines = [line.strip() for line in text]

        for each_folder in os.listdir(self.data_dir):
            parent_directory = each_folder[:15]
            if parent_directory in lines:
                temp_path = os.path.join(self.data_dir, each_folder, "tiles")

                if "frost" in os.listdir(temp_path):
                  for each_image in os.listdir(os.path.join(temp_path, "frost")):
                    image_path.append(os.path.join(temp_path, "frost", each_image))
                    labels.append(1)
                else:
                  for each_image in os.listdir(os.path.join(temp_path, "background")):
                    image_path.append(os.path.join(temp_path, "background", each_image))
                    labels.append(0)
        
        return image_path, labels 
        