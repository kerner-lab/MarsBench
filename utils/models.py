import torchvision.models as models
import torch


#SwinTransformer
def SwinTransformer(device, dataset_name, NUM_CLASSES, LR, run_type):
    # Selecting model paramters based on Run type
    if run_type=="Transfer_Learning" or run_type=="Finetuning":
        model=models.swin_b(weights='DEFAULT')

        if run_type=='Transfer_Learning':    
            for param in model.parameters():
                param.requires_grad = False

        if run_type=='Finetuning':
            for param in model.parameters():
                param.requires_grad = True
    
    if run_type=="Scratch":
        model=models.swin_b(weights=None)
        for param in model.parameters():
                param.requires_grad = True
                
    model.head = torch.nn.Linear(model.head.in_features, NUM_CLASSES)
    model.to(device)
    
    if dataset_name in ['DustyvsNonDusty']:
        criterion = torch.nn.BCEwithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=LR)

    return criterion, optimizer, model
    

#ResNet50
def ResNet50(device,dataset_name, NUM_CLASSES, LR, run_type):
    
    if run_type=="Transfer_Learning" or run_type=="Finetuning":
        model=models.resnet50(pretrained=True)
    
        if run_type=='Transfer_Learning':    
            for param in model.parameters():
                param.requires_grad = False

        if run_type=='Finetuning':
            for param in model.parameters():
                param.requires_grad = True
    
    if run_type=="Scratch":
        model=models.resnet50(pretrained=False)
        for param in model.parameters():
                param.requires_grad = True

    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(device)

    if dataset_name in ['DustyvsNonDusty']:
        criterion = torch.nn.BCEwithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=LR)

    return criterion, optimizer, model


#VIT16
def VIT16(device,dataset_name,NUM_CLASSES, LR, run_type):

    if run_type=="Transfer_Learning" or run_type=="Finetuning":
        model = models.vit_b_16(pretrained=True)
        
        if run_type=='Transfer_Learning':    
            for param in model.parameters():
                param.requires_grad = False

        if run_type=='Finetuning':
            for param in model.parameters():
                param.requires_grad = True
    
    if run_type=="Scratch":
        model=models.vit_b_16(pretrained=False)
        for param in model.parameters():
                param.requires_grad = True

    model.heads.head = torch.nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    model.to(device)

    if dataset_name in ['DustyvsNonDusty']:
        criterion = torch.nn.BCEwithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.heads.head.parameters(), lr=LR)
    
    return criterion, optimizer, model

#ResNet18
def ResNet18(device, dataset_name, NUM_CLASSES, LR, run_type):
    
    if run_type=="Transfer_Learning" or run_type=="Finetuning":
        model=models.resnet18(pretrained=True)
    
        if run_type=='Transfer_Learning':    
            for param in model.parameters():
                param.requires_grad = False

        if run_type=='Finetuning':
            for param in model.parameters():
                param.requires_grad = True
    
    if run_type=="Scratch":
        model=models.resnet18(pretrained=False)
        for param in model.parameters():
                param.requires_grad = True


    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(device)

    if dataset_name in ['DustyvsNonDusty']:
        criterion = torch.nn.BCEwithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=LR)

    return criterion, optimizer, model

#SqueezeNet1_1
def SqueezeNet(device, dataset_name, NUM_CLASSES, LR, run_type):
    
    if run_type=="Transfer_Learning" or run_type=="Finetuning":
        model=models.squeezenet1_1(pretrained=True)
    
        if run_type=='Transfer_Learning':    
            for param in model.parameters():
                param.requires_grad = False

        if run_type=='Finetuning':
            for param in model.parameters():
                param.requires_grad = True
    
    if run_type=="Scratch":
        model=models.squeezenet1_1(pretrained=False)
        for param in model.parameters():
                param.requires_grad = True


    model.classifier[1] = torch.nn.Conv2d(512, NUM_CLASSES, kernel_size=(1,1), stride=(1,1))

    model.num_classes = NUM_CLASSES
    model.to(device)

    if dataset_name in ['DustyvsNonDusty']:
        criterion = torch.nn.BCEwithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.classifier[1].parameters(), lr=LR)

    return criterion, optimizer, model


#InceptionV3
def InceptionV3(device, dataset_name, NUM_CLASSES, LR, run_type):
    
    if run_type=="Transfer_Learning" or run_type=="Finetuning":
        model=models.inception_v3(pretrained=True)
    
        if run_type=='Transfer_Learning':    
            for param in model.parameters():
                param.requires_grad = False

        if run_type=='Finetuning':
            for param in model.parameters():
                param.requires_grad = True
    
    if run_type=="Scratch":
        model=models.squeezenet1_1(pretrained=False)
        for param in model.parameters():
                param.requires_grad = True
                
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(device)

    if dataset_name in ['DustyvsNonDusty']:
        criterion = torch.nn.BCEwithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=LR)

    return criterion, optimizer, model