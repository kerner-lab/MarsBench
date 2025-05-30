"""
RetinaNet model implementation for object detection in Mars surface images.
"""


import logging
from functools import partial

import torch
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from .BaseDetectionModel import BaseDetectionModel

logger = logging.getLogger(__name__)


class RetinaNet(BaseDetectionModel):
    def __init__(self, cfg):
        super(RetinaNet, self).__init__(cfg)

    def _initialize_model(self):
        num_classes = self.cfg.data.num_classes
        pretrained = self.cfg.model.pretrained
        freeze_layers = self.cfg.model.freeze_layers

        if pretrained:
            weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        else:
            weights = None
        model = retinanet_resnet50_fpn_v2(weights=weights)
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32),
        )

        if freeze_layers and not pretrained:
            logger.warning("freeze_layers is set to True but model is not pretrained. Setting freeze_layers to False")
            freeze_layers = False

        if pretrained and freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.backbone.fpn.parameters():
                param.requires_grad = True
            for param in model.head.parameters():
                param.requires_grad = True
            logger.info("Froze model layers, keeping fpn and head trainable")
        else:
            for param in model.parameters():
                param.requires_grad = True
            if pretrained:
                logger.info("Using pretrained weights with all layers trainable")
            else:
                logger.info("Training from scratch with all layers trainable")

        return model
