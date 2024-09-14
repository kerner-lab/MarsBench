import random
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from skimage import io
import numpy as np
import torchvision.models as models
import warnings
import datetime
import sys
import wandb
from data.data import DeepMars_Landmark, DeepMars_Surface, DoMars16k, MarsDatasetLandmark, MartianFrost, MarsDatasetSurface
from utils.train import train

warnings.filterwarnings("ignore")

#System arguments
wandb_run = sys.argv[1]
model_name = str(sys.argv[2])
dataset_name= str(sys.argv[3])

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
N_EPOCHS = 100
IMAGE_SIZE = (224,224)

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

    train_dataset = DoMars16k(data_dir = TRAIN_DIR, transform = transform)
    val_dataset = DoMars16k(data_dir = VALID_DIR, transform = target_transform)
    test_dataset = DoMars16k(data_dir = TEST_DIR, transform = target_transform)

#Define variables for Mars Content Classification Landmark
if dataset_name == "MarsDatasetLandmark":
    NUM_CLASSES = 8
    DATA_DIR = '/data/hkerner/MarsBench/Datasets/Mars_Image_Cont_Class_Landmark/hirise-map-proj-v3_2/map-proj-v3_2/'
    TXT_FILE = '/data/hkerner/MarsBench/Datasets/Mars_Image_Cont_Class_Landmark/hirise-map-proj-v3_2/labels-map-proj_v3_2_train_val_test.txt' 

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

    train_dataset = MarsDatasetSurface(data_dir = DATA_DIR, transform = transform, txt_file = TRAIN_TXT)
    val_dataset = MarsDatasetSurface(data_dir = DATA_DIR, transform = target_transform, txt_file = VAL_TXT)
    test_dataset = MarsDatasetSurface(data_dir = DATA_DIR, transform = target_transform, txt_file = TEST_TXT)

#Define variables DeepMars Landmark
if dataset_name  == "DeepMars_Landmark":
    NUM_CLASSES = 6
    DATA_DIR = '/data/hkerner/MarsBench/Datasets/DeepMars_Landmark/map-proj/'
    LABEL_TXT = '/data/hkerner/MarsBench/Datasets/DeepMars_Landmark/labels-map-proj.txt'

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


# Different Model definitions with specific conditions
#Swin Transformer
if model_name=='SwinTransformer':

    model=models.swin_b(weights='DEFAULT')
    
    for param in model.parameters():
        param.requires_grad = False

    model.head = torch.nn.Linear(model.head.in_features, NUM_CLASSES)
    model.to(device)
    
    if dataset_name in ['DustyvsNonDusty']:
        criterion = torch.nn.BCEwithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=LR)
    

#ResNet50
if model_name=='ResNet50':

    model=models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(device)

    if dataset_name in ['DustyvsNonDusty']:
        criterion = torch.nn.BCEwithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=LR)

#VIT16
if model_name=='VIT16':
    model = models.vit_b_16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    model.heads.head = torch.nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    model.to(device)

    if dataset_name in ['DustyvsNonDusty']:
        criterion = torch.nn.BCEwithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.heads.head.parameters(), lr=LR)


#Call Training Function
print("Training Begins")
train(model, train_data_loader, val_data_loader, test_data_loader ,criterion, optimizer, device, len(train_dataset), len(val_dataset), len(test_dataset) ,num_epochs=N_EPOCHS)
print('\n')
print("Training Ends")
