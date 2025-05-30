"""
Swin Transformer model implementation for Mars surface image classification.
"""
import logging

from torch import nn
from torchvision.models import Swin_V2_B_Weights
from torchvision.models import swin_v2_b

from .BaseClassificationModel import BaseClassificationModel

logger = logging.getLogger(__name__)


class SwinTransformer(BaseClassificationModel):
    def __init__(self, cfg):
        super(SwinTransformer, self).__init__(cfg)

    def _initialize_model(self):
        num_classes = self.cfg.data.num_classes
        pretrained = self.cfg.model.pretrained
        freeze_layers = self.cfg.model.freeze_layers

        if pretrained:
            weights = Swin_V2_B_Weights.DEFAULT
        else:
            weights = None

        model = swin_v2_b(weights=weights)

        num_features = model.head.in_features
        model.head = nn.Linear(num_features, num_classes)

        if freeze_layers and not pretrained:
            logger.warning("freeze_layers is set to True but model is not pretrained. Setting freeze_layers to False")
            freeze_layers = False

        if pretrained and freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
            logger.info("Froze transformer layers, keeping final classifier trainable")
        else:
            for param in model.parameters():
                param.requires_grad = True
            logger.info("All layers are trainable")

        return model
