import logging

import torch
import torch.nn as nn

from marsbench.utils.detr_criterion import SetCriterion
from marsbench.utils.detr_matcher import HungarianMatcher

from .BaseDetectionModel import BaseDetectionModel

logger = logging.getLogger(__name__)


class DETR(BaseDetectionModel):
    def __init__(self, cfg):
        super(DETR, self).__init__(cfg)
        num_classes = self.cfg.data.num_classes

        self.weight_dict = self.cfg.model.loss_weight_dict
        self.losses = self.cfg.model.loss_types
        NULL_CLASS_COEF = self.cfg.model.null_class_coef

        matcher = HungarianMatcher()
        self.criterion = SetCriterion(
            num_classes=num_classes - 1,
            matcher=matcher,
            weight_dict=self.weight_dict,
            eos_coef=NULL_CLASS_COEF,
            losses=self.losses,
        )

    def _initialize_model(self):
        num_classes = self.cfg.data.num_classes
        pretrained = self.cfg.model.pretrained
        freeze_layers = self.cfg.model.freeze_layers
        num_queries = self.cfg.model.num_queries

        model = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=pretrained)
        model.iou = self.cfg.model.iou
        in_features = model.class_embed.in_features
        model.class_embed = nn.Linear(in_features=in_features, out_features=num_classes)
        model.num_queries = num_queries

        if num_queries != 100:
            print("Warning: Changing query count requires full reinitialization of query embeddings!")
            model.num_queries = num_queries
            model.query_embed = nn.Embedding(num_queries, 256)

        if freeze_layers and not pretrained:
            logger.warning("freeze_layers is set to True but model is not pretrained. Setting freeze_layers to False")
            freeze_layers = False

        if pretrained and freeze_layers:
            for param in model.backbone.parameters():
                param.requires_grad = False

            for param in model.class_embed.parameters():
                param.requires_grad = True
            for param in model.bbox_embed.parameters():
                param.requires_grad = True
            logger.info("Froze model parameters, keeping class and bbox embedding layer trainable")

            if num_queries != 100:
                for param in model.query_embed.parameters():
                    param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True
            if pretrained:
                logger.info("Using pretrained weights with all layers trainable")
            else:
                logger.info("Training from scratch with all layers trainable")

        return model

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.DEVICE)

        outputs = self(images)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        self._log_metrics("train", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.DEVICE)
        outputs = self(images)

        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        print(f"Validation Loss: {total_loss}")

    def test_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.DEVICE)
        outputs = self(images)

        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        print(f"Test Loss: {total_loss}")
