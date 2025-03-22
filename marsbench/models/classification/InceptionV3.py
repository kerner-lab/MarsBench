"""
InceptionV3 model implementation for Mars surface image classification.
"""

import logging

from torch import nn
from torchvision.models import Inception_V3_Weights
from torchvision.models import inception_v3

from .BaseClassificationModel import BaseClassificationModel

logger = logging.getLogger(__name__)


class InceptionV3(BaseClassificationModel):
    def __init__(self, cfg):
        super(InceptionV3, self).__init__(cfg)

    def _initialize_model(self):
        num_classes = self.cfg.data.num_classes
        pretrained = self.cfg.model.pretrained
        freeze_layers = self.cfg.model.freeze_layers
        if pretrained:
            weights = Inception_V3_Weights.DEFAULT
        else:
            weights = None
        model = inception_v3(weights=weights, aux_logits=True)

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

        # Optionally replace auxiliary classifier
        num_aux_features = model.AuxLogits.fc.in_features  # type: ignore
        model.AuxLogits.fc = nn.Linear(num_aux_features, num_classes)  # type: ignore

        if freeze_layers and not pretrained:
            logger.warning("freeze_layers is set to True but model is not pretrained. Setting freeze_layers to False")
            freeze_layers = False

        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True

        return model

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)

        loss_main = self.criterion(outputs.logits, labels)
        loss_aux = self.criterion(outputs.aux_logits, labels)
        loss = loss_main + 0.4 * loss_aux  # Weighted sum as per the original paper

        acc = self._calculate_accuracy(outputs.logits, labels)

        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)

        # eval state, we use only recieve logits
        loss = self.criterion(outputs, labels)
        acc = self._calculate_accuracy(outputs, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)

        loss = self.criterion(outputs, labels)
        acc = self._calculate_accuracy(outputs, labels)

        self.log("test_loss", loss)
        self.log("test_acc", acc)
