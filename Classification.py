import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
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
wandb_id= sys.argv[2]

#Wandb Stuff
wandb.login(key="ee2f13f7fdb31a577bcdc759c68c3c0a1ac2751d")
wandb.init(project='MarsBench', name=wandb_run, id=wandb_id)


#Seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

#Device initialisation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#DataSet
class DoMars16k(Dataset):
  def __init__(self, img_dir, transform=None,target_transform=None):
    self.transform = transform
    self.target_transform = target_transform
    self.img_dir = img_dir
    self.labels = []
    self.image_paths = []

    for label, class_dir in enumerate(os.listdir(img_dir)):
      class_dir_path = os.path.join(img_dir, class_dir)
      
      for image in os.listdir(class_dir_path):
        if image.endswith(('jpg','png','jpeg')):
          image_path = os.path.join(class_dir_path,image)
          self.image_paths.append(image_path)
          self.labels.append(label)
    
  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self,idx):
    #image = Image.open(self.image_paths[idx]).convert('RGB')
    image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    label = self.labels[idx]

    if self.transform:
      image = self.transform(image)

    if self.target_transform:
      image=self.target_transform(image)
    
    return image,label
  
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
NUM_CLASSES = 15
BATCH_SIZE = 16
LR = 0.0001
MOMENTUM = 0.9
N_EPOCHS = 30

#Define Directories and other variables
TRAIN_DIR = '/home/kkasodek/MarsBench/datasets/data/train/'
VALID_DIR = '/home/kkasodek/MarsBench/datasets/data/val/'
TEST_DIR = '/home/kkasodek/MarsBench/datasets/data/test/'
IMAGE_SIZE = (224,224)


print("Execution Date-Time: ",datetime.datetime.now())
print("RESNET50 with DoMars16k, Normalized using ImageNet data and no Uniform Random Sampling")
#print("VIT with DoMars16k, Normalized using ImageNet data and no Uniform Random Sampling")
#print("SWIN-Transformer with DoMars16k, Normalized using ImageNet data and no Uniform Random Sampling")

#Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

target_transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


#Dataset Created
train_dataset = DoMars16k(img_dir=TRAIN_DIR,transform=transform)
val_dataset = DoMars16k(img_dir=VALID_DIR, transform=target_transform)


#Creating weights for uniform sampling
label_count = np.bincount(train_dataset.labels)
inverse_weights = 1./label_count
sample_weights = inverse_weights[train_dataset.labels]

#Uniform sampler on the basis of inverse counts
uniform_sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights,num_samples=len(sample_weights),replacement=True)

#Using Boot Sampler
num_samples = 10
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
# Resnet
# model=models.resnet50(pretrained=True)
# VIT
model=models.vit_b_16(pretrained=True)
print(model)
# Freeze all the pre-trained layers
for param in model.parameters():
  param.requires_grad = False

# Modify the last layer of the model
num_classes = NUM_CLASSES
# model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# VIT
model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
model.to(device)


# 
model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()

# Fine-tune the last layer for a few epochs
optimizer = torch.optim.Adam(model.heads.head.parameters(), lr=LR)
print("Training Begins")
train(model, train_data_loader, val_data_loader, criterion, optimizer, num_epochs=N_EPOCHS)
print('\n')
print("Training Ends")
print('\n')
print('\n')
print('\n')
print('\n')



