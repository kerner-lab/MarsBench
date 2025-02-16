import torch.nn as nn
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models import ResNet34_Weights
from torchvision.models import resnet34
from torchvision.models.detection.ssd import SSD as torch_SSD
from torchvision.models.detection.ssd import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSDHead

from .BaseDetectionModel import BaseDetectionModel


class SSD(BaseDetectionModel):
    def __init__(self, cfg):
        super(SSD, self).__init__(cfg)
        self.metrics = MeanAveragePrecision(iou_type="bbox")

    def _initialize_model(self):
        num_classes = self.cfg.data.num_classes + 1
        nms = self.cfg.model.detection.nms
        size = tuple(self.cfg.transforms.image_size)
        pretrained = self.cfg.model.detection.pretrained
        freeze_layers = self.cfg.model.detection.freeze_layers

        if pretrained:
            weights = ResNet34_Weights.DEFAULT
        else:
            weights = None

        model_backbone = resnet34(weights=weights)

        conv1 = model_backbone.conv1
        bn1 = model_backbone.bn1
        relu = model_backbone.relu
        max_pool = model_backbone.maxpool
        layer1 = model_backbone.layer1
        layer2 = model_backbone.layer2
        layer3 = model_backbone.layer3
        layer4 = model_backbone.layer4
        backbone = nn.Sequential(
            conv1, bn1, relu, max_pool, layer1, layer2, layer3, layer4
        )
        out_channels = [512, 512, 512, 512, 512, 512]
        anchor_generator = DefaultBoxGenerator(
            [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        )
        num_anchors = anchor_generator.num_anchors_per_location()
        head = SSDHead(out_channels, num_anchors, num_classes)

        model = torch_SSD(
            backbone=backbone,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
            size=size,
            head=head,
            nms_thresh=nms,
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
