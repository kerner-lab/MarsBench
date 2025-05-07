"""
InceptionV3 model implementation for Mars surface image classification.
"""

import logging

import torch.nn.functional as F
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

    def _common_step(self, batch, batch_idx, metrics, phase):
        imgs, gt = batch
        if phase == "train":
            outputs = self(imgs)
            logits = outputs.logits
            aux_logits = outputs.aux_logits
        else:
            logits = self(imgs)
            aux_logits = None

        if self.cfg.data.subtask == "binary":
            logits = logits.squeeze(-1)
            aux_logits = aux_logits.squeeze(-1) if aux_logits is not None else None

        if phase == "train":
            # Weighted sum as per the original paper
            loss_main = self.criterion(logits, gt)
            loss_aux = self.criterion(aux_logits, gt)
            loss = loss_main + 0.4 * loss_aux
        else:
            loss = self.criterion(logits, gt)

        metrics.update(logits.detach(), gt.detach().long())
        self.log(
            f"{phase}/loss",  # required for early stopping
            loss,
            on_step=(phase == "train"),
            on_epoch=True,
            prog_bar=True,
        )

        if (
            (self.current_epoch % self.vis_every == 0 or self.current_epoch == self.trainer.max_epochs - 1)
            and batch_idx == 0
            and self.current_epoch != 0
        ):
            if self.cfg.data.subtask == "multiclass":
                probs = F.softmax(logits, dim=1)
                preds = probs.argmax(1)
            else:
                probs = F.sigmoid(logits)
                preds = (probs > 0.5).long()
            self._store_vis(phase, imgs, gt.long(), probs, preds)
        return loss
