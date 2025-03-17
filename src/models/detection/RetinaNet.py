from functools import partial

import torch
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from .BaseDetectionModel import BaseDetectionModel


class RetinaNet(BaseDetectionModel):
    def __init__(self, cfg):
        super(RetinaNet, self).__init__(cfg)
        self.metrics = MeanAveragePrecision(iou_type="bbox")

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
            print(
                "freeze_layers is set to True but model is not pretrained. Setting freeze_layers to False"
            )
            freeze_layers = False

        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True

        return model

    def _calculate_metrics(self, outputs, targets):
        for output, target in zip(outputs, targets):
            targets_dict = [
                {
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": target["labels"].detach().cpu(),
                }
            ]
            preds_dict = [
                {
                    "boxes": output["boxes"].detach().cpu(),
                    "labels": output["labels"].detach().cpu(),
                    "scores": output["scores"].detach().cpu(),
                }
            ]

        self.metrics.reset()
        self.metrics.update(preds_dict, targets_dict)
        metric_summary = self.metrics.compute()
        return metric_summary

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.DEVICE)
        outputs = self(images)

        if self.metrics:
            metric_summary = self._calculate_metrics(outputs, targets)
            print(f"validation metrics: {metric_summary}")

    def test_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.DEVICE)
        outputs = self(images)

        if self.metrics:
            metric_summary = self._calculate_metrics(outputs, targets)
            print(f"test metrics: {metric_summary}")
