import warnings

from torch import nn
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet18

from .BaseClassificationModel import BaseClassificationModel


class ResNet18(BaseClassificationModel):
    def __init__(self, cfg):
        super(ResNet18, self).__init__(cfg)

    def _initialize_model(self):
        num_classes = self.cfg.data.num_classes
        pretrained = self.cfg.model.classification.pretrained
        freeze_layers = self.cfg.model.classification.freeze_layers

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        model = resnet18(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

        if freeze_layers and not pretrained:
            warnings.warn(
                "freeze_layers is set to True but model is not pretrained. Setting freeze_layers to False"
            )
            freeze_layers = False

        if pretrained and freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True

        return model
