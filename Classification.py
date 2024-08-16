import random
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from skimage import io
import numpy as np
import torchvision.models as models
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from PIL import Image
import warnings
import datetime
import wandb
import psutil
import sys
import cv2

warnings.filterwarnings("ignore")

wandb_run=sys.argv[1]
#wandb_id= sys.argv[2]

#Wandb Stuff
wandb.login(key="ee2f13f7fdb31a577bcdc759c68c3c0a1ac2751d")
wandb.init(project='MarsBench', name=wandb_run)

#Seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

#Device initialisation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cpu'
# Base Dataset Class
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
class MarsDataset(CustomDataset): 
    def __init__(self, data_dir, transform, txt_file, split_type):
        self.text_file = txt_file
        self.split_type = split_type
        super(MarsDataset, self).__init__(data_dir, transform)
       

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
               labels.append(int(label))
        
        return image_path, labels
    
    
class MartianFrostDataset(CustomDataset):
    def _getdata(self):
        image_paths = []
        labels = []
        
        # Walk through the directories and collect image and label file paths
        for subframe_dir in os.listdir(self.data_dir):
            subframe_path = os.path.join(self.data_dir, subframe_dir)

            if not os.path.isdir(subframe_path):
                continue

            tiles_dir = os.path.join(subframe_path, "tiles")
            labels_dir = os.path.join(subframe_path, "labels")
            
            for frost_type in os.listdir(tiles_dir):
                tile_frost_dir = os.path.join(tiles_dir, frost_type)
                label_frost_dir = os.path.join(labels_dir, frost_type)

                # Skip non-directory files
                if not os.path.isdir(tile_frost_dir):
                    continue

                for img_file in os.listdir(tile_frost_dir):
                    img_path = os.path.join(tile_frost_dir, img_file)
                    label_file = img_file.replace(".png", ".json")
                    label_path = os.path.join(label_frost_dir, label_file)

                    # Skip non-png files or missing labels
                    if not img_file.endswith('.png') or not os.path.exists(label_path):
                        continue
                    
                    # Store paths
                    print(img_path)
                    image_paths.append(img_path)

                    # Load the label from the corresponding JSON file
                    with open(label_path, 'r') as f:
                        label_data = json.load(f)
                    label = label_data.get('frost', 0)
                    labels.append(label)

        return image_paths, labels
 

# Wandb log table  
def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(20)])
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table":table}, commit=False)
  
#Bootstrap sampler use if required
# def bootstrap_sampler(dataset,num_samples):
#     ind=np.random.choice(len(dataset),size=num_samples, replace=True)
#     return torch.utils.data.Subset(dataset,ind)


# Training Loop
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
  # Train the model for the specified number of epochs
  mem_info = psutil.virtual_memory()
  for epoch in range(num_epochs):
  # Set the model to train mode
    model.train()

    # Initialize the running loss and accuracy
    running_loss = 0.0
    running_corrects = 0
    wandb.log({"Memory Usage (MB)": mem_info.used / (1024 ** 2)}, step=epoch)
    # Iterate over the batches of the train loader
    for inputs, labels in train_loader:
        # Move the inputs and labels to the device
      inputs = inputs.to(device)
      labels = labels.to(device)
      # print(labels)

      # Zero the optimizer gradients
      optimizer.zero_grad()

      # Forward pass
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      loss = criterion(outputs, labels)

      # Backward pass and optimizer step
      loss.backward()
      optimizer.step()
      metric1 = {'train_step_loss':loss}
      wandb.log(metric1)
      # Update the running loss and accuracy
      running_loss += loss.item() * inputs.size(0)
      metric2={'train_running_loss':running_loss}
      running_corrects += torch.sum(preds == labels.data)

    # Calculate the train loss and accuracy

    train_loss = running_loss / len(train_dataset)
    train_acc = running_corrects.double() / len(train_dataset)

    # Set the model to evaluation mode
    model.eval()

    # Initialize the running loss and accuracy
    running_loss = 0.0
    running_corrects = 0

    # Iterate over the batches of the validation loader
    with torch.no_grad():
      for inputs, labels in val_loader:
        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # Update the running loss and accuracy
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    # Calculate the validation loss and accuracy
    val_loss = running_loss / len(val_dataset)
    val_acc = running_corrects.double() / len(val_dataset)
    metrics3 = {"training_loss":train_loss,'validation_loss':val_loss}
    metrics4 = {"training_acc":train_acc,'validation_acc':val_acc}
    wandb.log(metrics3)
    wandb.log(metrics4)
    # Print the epoch results
    print('Epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'
          .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))
    validate(model, val_loader)

  # Validation
def validate(model, val_loader):
  all_preds = []
  all_labels = []
  # Set the model to evaluation mode
  model.eval()

  # Iterate over the batches of the validation loader
  with torch.no_grad():
    for inputs, labels in val_loader:
      # Move the inputs and labels to the device
      inputs = inputs.to(device)
      labels = labels.to(device)

      # Forward pass
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)

      # Append current batch predictions and labels to lists
      all_preds.extend(preds.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

  # Calculate precision, recall, F1 score
  precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
  metrics4={"precision":precision,"recall":recall,"f1_score":f1_score}
  wandb.log(metrics4)
  # Print the epoch results
  print('val precision: {:.4f}, val recall: {:.4f}, val F1 score: {:.4f}'
        .format(precision, recall, f1_score))

#Main Code begins
#NUM_CLASSES = 15  #DoMars16k
NUM_CLASSES = 8   #Mars Content Classification Landmark
#NUM_CLASSES =  6   #DeepMars Landmark
#NUM_CLASSES = 24  #DeepMars Surface

BATCH_SIZE = 16
LR = 0.0001
MOMENTUM = 0.9
N_EPOCHS = 30
IMAGE_SIZE = (224,224)

#Define Directories and other variables for DoMars16K
#TRAIN_DIR = '/data/hkerner/MarsBench/Datasets/DoMars16K/data/train/'
#VALID_DIR = '/data/hkerner/MarsBench/Datasets/DoMars16K/data/val/'
#TEST_DIR = '/data/hkerner/MarsBench/Datasets/DoMars16K/data/test/'

#Define Directories and other variables for Mars Image Content Classification-Landmark
#DATA_DIR = '/data/hkerner/MarsBench/Datasets/Mars_Image_Cont_Class_Landmark/hirise-map-proj-v3_2/map-proj-v3_2/'
#TXT_FILE = '/data/hkerner/MarsBench/Datasets/Mars_Image_Cont_Class_Landmark/hirise-map-proj-v3_2/labels-map-proj_v3_2_train_val_test.txt' 

#Define Directories and other variables for Martian Frost Dataset
data_dir = '/data/hkerner/MarsBench/Datasets/Martian_Frost/data' 
  
#Define Directories and other variables for DeepMars Landmark 
# DATA_DIR = '/data/hkerner/MarsBench/Datasets/DeepMars_Landmark/map-proj/'
# LABEL_TXT = '/data/hkerner/MarsBench/Datasets/DeepMars_Landmark/labels-map-proj.txt'

# #Define Directories and other variables for DeepMars Surface 
# DATA_DIR = '/data/hkerner/MarsBench/Datasets/DeepMars_Surface/calibrated/'
# TRAIN_TXT = '/data/hkerner/MarsBench/Datasets/DeepMars_Surface/train-calibrated-shuffled.txt'
# VAL_TXT = '/data/hkerner/MarsBench/Datasets/DeepMars_Surface/val-calibrated-shuffled.txt'
# TEST_TXT = '/data/hkerner/MarsBench/Datasets/DeepMars_Surface/test-calibrated-shuffled.txt'


print("Execution Date-Time: ",datetime.datetime.now())
print("VIT-16 with Mars Content Classification, using 75:25 split  Normalized using ImageNet data and no Uniform Random Sampling")

#Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.25),
    transforms.RandomApply([transforms.RandomRotation((180,180))],p=0.25),
    transforms.RandomApply([transforms.RandomRotation((270,270))],p=0.25),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

target_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# # DeepMars Surface classification dataset created
# train_dataset = DeepMars_Surface(data_dir = DATA_DIR, transform = transform, txt_file = TRAIN_TXT)
# val_dataset = DeepMars_Surface(data_dir = DATA_DIR, transform = target_transform, txt_file = VAL_TXT)


## dataset creation for DeepMars Landmark
# dataset=DeepMars_Landmark(data_dir = DATA_DIR, txt_file = LABEL_TXT, transform = target_transform)
# train_size = int(0.7 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, test_size])


# # Mars Image content classification dataset created
# train_dataset = MarsDataset(data_dir = DATA_DIR,transform=transform, txt_file = TXT_FILE, split_type ='train')
# val_dataset = MarsDataset(data_dir = DATA_DIR,transform=target_transform, txt_file = TXT_FILE, split_type ='val')

##DoMars16k Dataset created
# train_dataset = DoMars16k(data_dir = TRAIN_DIR, transform = transform)
# val_dataset = DoMars16k(data_dir = VALID_DIR, transform = target_transform)

# Dataset Creation for Martian Frost
martian_frost_dataset = MartianFrostDataset(data_dir=data_dir, transform=transform)
train_dataset, val_dataset = split_dataset(dataset, split_ratio=0.2)

##Creating weights for uniform sampling
# label_count = np.bincount(train_dataset.labels)
# inverse_weights = 1./label_count
# sample_weights = inverse_weights[train_dataset.labels]

##Uniform sampler on the basis of inverse counts
#uniform_sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights,num_samples=len(sample_weights),replacement=True)

#Using Boot Sampler
#num_samples = 10
#bootstrap_datasets=[bootstrap_sampler(train_dataset, len(train_dataset)) for _ in range(num_samples)]

# Initializing DataLoader
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,sampler=None)
val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

mean = torch.zeros(3)
std = torch.zeros(3)
nb_samples = 0


# Iterate through the DataLoader
for data in train_data_loader:
  images, _ = data
  batch_samples = images.size(0)  # batch size (the last batch can have smaller size)
  images = images.view(batch_samples, images.size(1), -1)
  mean += images.mean(2).sum(0)
  std += images.std(2).sum(0)
  nb_samples += batch_samples

# Compute mean and std
mean /= nb_samples
std /= nb_samples

print(f"Mean: {mean}")
print(f"Std: {std}")


# Load the pre-trained Swin Transformer BASE model
#model = models.swin_b(pretrained=True)
# model=models.swin_b(weights='DEFAULT')

# ResNet
# model = models.resnet50(pretrained=True)

# VIT
model = models.vit_b_16(pretrained=True) 


# Freeze all the pre-trained layers
for param in model.parameters():
  param.requires_grad = False

# Modify the last layer of the model
num_classes = NUM_CLASSES
# model.head = torch.nn.Linear(model.head.in_features, num_classes)

# VIT
model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
model.to(device)

# RESNET
# model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
# model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()

# Fine-tune the last layer for a few epochs
# AdamW
optimizer = torch.optim.AdamW(model.heads.head.parameters(), lr=LR)

# Adam
#optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

print("Training Begins")
train(model, train_data_loader, val_data_loader, criterion, optimizer, num_epochs=N_EPOCHS)
print('\n')
print("Training Ends")
print('\n')
print('\n')
print('\n')
print('\n')



