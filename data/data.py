
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
    

class MartianFrostDataset(Dataset):
    def __init__(self, data_dir, transform, split_type):
        self.data_dir=data_dir
        self.images = []
        self.labels = []
        self.split_type=split_type
        self.transform = transform
        data_df = self._getdata()
        

        for _, row in data_df.iterrows():
            image_path = row["filepath"]
            self.images.append(image_path)
            self.labels.append(row["label"])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        image_path = self.images[idx]
        label = self.labels[idx]
        filename = image_path.split("/")[-1]

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # image = np.expand_dims(image, axis=0)
        image = np.stack((image,)*3, axis=-1)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if self.transform:
            image=self.transform(image)

        return image, label, filename.split("/")[-1]
    
    def _getdata(self):
        data_list = []

        if self.split_type=="train":
            with open(os.path.join(self.data_dir, "train_source_images.txt")) as f:
                lines = f.readlines()
                for each_line in lines:
                    data_list.append(each_line[:-1])

        if self.split_type=="val":
            with open(os.path.join(self.data_dir, "val_source_images.txt")) as f:
                lines = f.readlines()
                for each_line in lines:
                    data_list.append(each_line[:-1])

        if self.split_type=="test":
            with open(os.path.join(self.data_dir, "test_source_images.txt")) as f:
                lines = f.readlines()
                for each_line in lines:
                    data_list.append(each_line[:-1])

        data_folder = os.path.join(self.data_dir, "data")

        filepath_list, feature_list, set_list = [], [], []

        for each_folder in os.listdir(data_folder):

            if each_folder == ".DS_Store":
                continue

            class_name = os.listdir(os.path.join(data_folder, each_folder, "tiles"))[0]
            for each_file in os.listdir(os.path.join(data_folder, each_folder, "tiles", class_name)):
                filepath_list.append(os.path.join(data_folder, each_folder, "tiles", class_name, each_file))
                feature_list.append(class_name)
                set_list.append(self.split_type)

        data_dict = {"filepath": filepath_list, "feature": feature_list, "set": set_list}
        data_df = pd.DataFrame(data_dict)

        data_df = data_df.drop(data_df[data_df["filepath"]==".DS_Store"].index)

        le = LabelEncoder()
        le.fit(data_df["feature"])

        data_df["label"] = le.transform(data_df["feature"])

        return data_df

        