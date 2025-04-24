"""
ResNet101 model implementation for Mars surface image classification.
"""

import logging

from torch import nn
from torchvision.models import ResNet101_Weights
from torchvision.models import resnet101

from .BaseClassificationModel import BaseClassificationModel

logger = logging.getLogger(__name__)


class ResNet101(BaseClassificationModel):
    def __init__(self, cfg):
        super(ResNet101, self).__init__(cfg)

    def _initialize_model(self):
        num_classes = self.cfg.data.num_classes
        pretrained = self.cfg.model.pretrained
        freeze_layers = self.cfg.model.freeze_layers

        if pretrained:
            weights = ResNet101_Weights.DEFAULT
        else:
            weights = None

        model = resnet101(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

        if freeze_layers and not pretrained:
            logger.warning("freeze_layers is set to True but model is not pretrained. Setting freeze_layers to False")
            freeze_layers = False

        if pretrained and freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
            logger.info("Froze backbone layers, keeping final classifier trainable")
        else:
            for param in model.parameters():
                param.requires_grad = True
            logger.info("All layers are trainable")

        return model
