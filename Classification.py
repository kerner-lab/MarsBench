import random
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from skimage import io
import numpy as np
#import torchvision.models as models
import warnings
import datetime
import sys
import wandb
from data.data import DeepMars_Landmark, DeepMars_Surface, DoMars16k, MarsDatasetLandmark, MartianFrost, MarsDatasetSurface
from utils.train import train
from utils import models
from utils import val
import os

warnings.filterwarnings("ignore")

#System arguments
wandb_run = sys.argv[1]
model_name = str(sys.argv[2])
dataset_name = str(sys.argv[3])
run_type = str(sys.argv[4])

#Wandb Stuff
wandb.login(key="ee2f13f7fdb31a577bcdc759c68c3c0a1ac2751d")
wandb.init(project='MarsBench', name=wandb_run)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Seed
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

#Main Code begins
BATCH_SIZE = 16
LR = 0.0001
MOMENTUM = 0.9
N_EPOCHS = 5
IMAGE_SIZE = (224,224)
early_stopping_tolerance = 3
early_stopping_threshold = 0.03

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


#Define variables for DoMars16k
if dataset_name == "DoMars16k":
    NUM_CLASSES = 15
    TRAIN_DIR = '/data/hkerner/MarsBench/Datasets/DoMars16K/data/train/'
    VALID_DIR = '/data/hkerner/MarsBench/Datasets/DoMars16K/data/val/'
    TEST_DIR = '/data/hkerner/MarsBench/Datasets/DoMars16K/data/test/'
    MODEL_LOC = f'/data/hkerner/MarsBench/Models/DoMars16K/{model_name}/'
    if not os.path.isdir(MODEL_LOC):
        os.makedirs(MODEL_LOC)


    train_dataset = DoMars16k(data_dir = TRAIN_DIR, transform = transform)
    val_dataset = DoMars16k(data_dir = VALID_DIR, transform = target_transform)
    test_dataset = DoMars16k(data_dir = TEST_DIR, transform = target_transform)

#Define variables for Mars Content Classification Landmark
if dataset_name == "MarsDatasetLandmark":
    NUM_CLASSES = 8
    DATA_DIR = '/data/hkerner/MarsBench/Datasets/Mars_Image_Cont_Class_Landmark/hirise-map-proj-v3_2/map-proj-v3_2/'
    TXT_FILE = '/data/hkerner/MarsBench/Datasets/Mars_Image_Cont_Class_Landmark/hirise-map-proj-v3_2/labels-map-proj_v3_2_train_val_test.txt' 
    MODEL_LOC = f'/data/hkerner/MarsBench/Models/MarsDatasetLandmark/{model_name}/'
    if not os.path.isdir(MODEL_LOC):
        os.makedirs(MODEL_LOC)


    train_dataset = MarsDatasetLandmark(data_dir = DATA_DIR,transform=transform, txt_file = TXT_FILE, split_type ='train')
    val_dataset = MarsDatasetLandmark(data_dir = DATA_DIR,transform=target_transform, txt_file = TXT_FILE, split_type ='val')
    test_dataset = MarsDatasetLandmark(data_dir = DATA_DIR,transform=target_transform, txt_file = TXT_FILE, split_type ='test')

#Define Variables for Mars Content Classification Surface
if dataset_name == "MarsDatasetSurface":
    NUM_CLASSES = 19
    DATA_DIR = '/data/hkerner/MarsBench/Datasets/Mars_Image_Cont_Class_Surface/msl-labeled-data-set-v2.1/images/'
    TRAIN_TXT = '/data/hkerner/MarsBench/Datasets/Mars_Image_Cont_Class_Surface/msl-labeled-data-set-v2.1/train-set-v2.1.txt'
    VAL_TXT = '/data/hkerner/MarsBench/Datasets/Mars_Image_Cont_Class_Surface/msl-labeled-data-set-v2.1/val-set-v2.1.txt'
    TEST_TXT = '/data/hkerner/MarsBench/Datasets/Mars_Image_Cont_Class_Surface/msl-labeled-data-set-v2.1/test-set-v2.1.txt'
    MODEL_LOC = f'/data/hkerner/MarsBench/Models/MarsDatasetSurface/{model_name}/'
    if not os.path.isdir(MODEL_LOC):
        os.makedirs(MODEL_LOC)

    train_dataset = MarsDatasetSurface(data_dir = DATA_DIR, transform = transform, txt_file = TRAIN_TXT)
    val_dataset = MarsDatasetSurface(data_dir = DATA_DIR, transform = target_transform, txt_file = VAL_TXT)
    test_dataset = MarsDatasetSurface(data_dir = DATA_DIR, transform = target_transform, txt_file = TEST_TXT)

#Define variables DeepMars Landmark
if dataset_name  == "DeepMars_Landmark":
    NUM_CLASSES = 6
    DATA_DIR = '/data/hkerner/MarsBench/Datasets/DeepMars_Landmark/map-proj/'
    LABEL_TXT = '/data/hkerner/MarsBench/Datasets/DeepMars_Landmark/labels-map-proj.txt'
    MODEL_LOC = f'/data/hkerner/MarsBench/Models/DeepMars_Landmark/{model_name}/'
    if not os.path.isdir(MODEL_LOC):
        os.makedirs(MODEL_LOC)

    dataset=DeepMars_Landmark(data_dir = DATA_DIR, txt_file = LABEL_TXT, transform = target_transform)
    train_size = int(0.6 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, other_dataset = random_split(dataset, [train_size, test_size])
    val_dataset, test_dataset = random_split(other_dataset, [0.5 * test_size, 0.5 * test_size])

#Define variables for DeepMars Surface
if dataset_name == "DeepMars_Surface":
    NUM_CLASSES = 24
    DATA_DIR = '/data/hkerner/MarsBench/Datasets/DeepMars_Surface/'
    TRAIN_TXT = '/data/hkerner/MarsBench/Datasets/DeepMars_Surface/train-calibrated-shuffled.txt'
    VAL_TXT = '/data/hkerner/MarsBench/Datasets/DeepMars_Surface/val-calibrated-shuffled.txt'
    TEST_TXT = '/data/hkerner/MarsBench/Datasets/DeepMars_Surface/test-calibrated-shuffled.txt'
    MODEL_LOC = f'/data/hkerner/MarsBench/Models/DeepMars_Surface/{model_name}/'
    if not os.path.isdir(MODEL_LOC):
        os.makedirs(MODEL_LOC)

    train_dataset = DeepMars_Surface(data_dir = DATA_DIR, transform = transform, txt_file = TRAIN_TXT)
    val_dataset = DeepMars_Surface(data_dir = DATA_DIR, transform = target_transform, txt_file = VAL_TXT)
    test_dataset = DeepMars_Surface(data_dir = DATA_DIR, transform = target_transform, txt_file = TEST_TXT)

#Define variables for Martian Frost
if dataset_name == "MartianFrost":
    NUM_CLASSES = 2
    DATA_DIR = "/data/hkerner/MarsBench/Datasets/Martian_Frost/data"
    TRAIN_TXT = "/data/hkerner/MarsBench/Datasets/Martian_Frost/train_source_images.txt"
    VAL_TXT = "/data/hkerner/MarsBench/Datasets/Martian_Frost/val_source_images.txt"
    TEST_TXT = "/data/hkerner/MarsBench/Datasets/Martian_Frost/test_source_images.txt"
    MODEL_LOC = f'/data/hkerner/MarsBench/Models/Martian_Frost/{model_name}/'
    if not os.path.exists(MODEL_LOC):
        os.makedirs(MODEL_LOC)

    train_dataset = MartianFrost(data_dir=DATA_DIR, transform=transform, txt_file=TRAIN_TXT)
    val_dataset = MartianFrost(data_dir=DATA_DIR, transform=transform, txt_file=VAL_TXT)
    test_dataset = MartianFrost(data_dir=DATA_DIR, transform=transform, txt_file=TEST_TXT)
    




print("Execution Date-Time: ",datetime.datetime.now())
print(f"{model_name} with {dataset_name}, Normalized using ImageNet data and no Uniform Random Sampling")


# #Using Boot Sampler
# num_samples = 10
# bootstrap_datasets=[bootstrap_sampler(train_dataset, len(train_dataset)) for _ in range(num_samples)]

# Initializing DataLoader
train_data_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4,sampler = None)
val_data_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)
test_data_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)


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


# Models  with specific conditions
#Swin Transformer
if model_name == 'SwinTransformer':
    criterion, optimizer, model = models.SwinTransformer(device, dataset_name, NUM_CLASSES, LR, run_type)
    
#ResNet50
if model_name == 'ResNet50':
    criterion, optimizer, model = models.ResNet50(device, dataset_name, NUM_CLASSES, LR, run_type)

#VIT16
if model_name == 'VIT16':
    criterion,optimizer, model = models.VIT16(device, dataset_name, NUM_CLASSES, LR, run_type)

#InceptionV3
if model_name == 'InceptionV3':
    criterion,optimizer, model = models.InceptionV3(device, dataset_name, NUM_CLASSES, LR, run_type)

#SqueezeNet
if model_name == 'SqueezeNet':
    criterion,optimizer, model = models.SqueezeNet(device, dataset_name, NUM_CLASSES, LR, run_type)

#Resnet18
if model_name == "Resnet18":
    criterion,optimizer, model = models.ResNet18(device, dataset_name, NUM_CLASSES, LR, run_type)

#Call Training Function
print("Training Begins")
train(model, train_data_loader, val_data_loader, criterion, optimizer, device, len(train_dataset), 
      len(val_dataset), MODEL_LOC, early_stopping_tolerance, early_stopping_threshold, num_epochs = N_EPOCHS)
print('\n')
print("Training Ends")


print("Testing Begins")
# Use best saved model for Testing
saved_model =torch.load(MODEL_LOC + 'best_model.pth')
saved_model = saved_model.to(device)

# Call test related functions
test_metrics1 = val.test(saved_model, test_data_loader, device, len(test_dataset), criterion)
test_precision, test_recall, test_F1_Score = val.validate(saved_model, test_data_loader, device)

test_metrics2 = {'test_precision': test_precision, 'test_recall': test_recall, 'test_F1_Score': test_F1_Score}

print('test loss: {:.4f}, train acc: {:.4f}'
          .format(test_metrics1['test_loss'], test_metrics1['test_acc']))
print("test_metrics2- ", test_metrics2)

print('\n')
print("Testing Ends")

wandb.log(test_metrics1)
wandb.log(test_metrics2)
wandb.finish()



