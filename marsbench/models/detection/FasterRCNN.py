"""
FasterRCNN model implementation for object detection in Mars surface images.
"""

import logging

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .BaseDetectionModel import BaseDetectionModel

logger = logging.getLogger(__name__)


class FasterRCNN(BaseDetectionModel):
    def __init__(self, cfg):
        super(FasterRCNN, self).__init__(cfg)

    def _initialize_model(self):
        num_classes = self.cfg.data.num_classes
        pretrained = self.cfg.model.pretrained
        freeze_layers = self.cfg.model.freeze_layers

        if pretrained:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        else:
            weights = None
        model = fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        if freeze_layers and not pretrained:
            logger.warning("freeze_layers is set to True but model is not pretrained. Setting freeze_layers to False")
            freeze_layers = False

        if pretrained and freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.roi_heads.parameters():
                param.requires_grad = True
            for param in model.rpn.parameters():
                param.requires_grad = True
            logger.info("Froze model parameter, keeping roi heads and rpn trainable")
        else:
            for param in model.parameters():
                param.requires_grad = True
            if pretrained:
                logger.info("Using pretrained weights with all layers trainable")
            else:
                logger.info("Training from scratch with all layers trainable")

        return model
