import logging

from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .BaseDetectionModel import BaseDetectionModel

logger = logging.getLogger(__name__)


class FasterRCNN(BaseDetectionModel):
    def __init__(self, cfg):
        super(FasterRCNN, self).__init__(cfg)
        self.metrics = MeanAveragePrecision(iou_type="bbox")

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
            logger.warning(
                "freeze_layers is set to True but model is not pretrained. Setting freeze_layers to False"
            )
            freeze_layers = False

        if pretrained and freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.roi_heads.box_predictor.parameters():
                param.requires_grad = True
            logger.info("Froze model parameter, keeping roi head bbox predictor")
        else:
            for param in model.parameters():
                param.requires_grad = True
            if pretrained:
                logger.info("Using pretrained weights with all layers trainable")
            else:
                logger.info("Training from scratch with all layers trainable")

        return model

    def _calculate_metrics(self, outputs, targets):
        targets_list = []
        preds_list = []
        for output, target in zip(outputs, targets):
            targets_dict = {
                "boxes": target["boxes"].detach().cpu(),
                "labels": target["labels"].detach().cpu(),
            }
            preds_dict = {
                "boxes": output["boxes"].detach().cpu(),
                "labels": output["labels"].detach().cpu(),
                "scores": output["scores"].detach().cpu(),
            }
            targets_list.append(targets_dict)
            preds_list.append(preds_dict)

        self.metrics.reset()
        self.metrics.update(preds_list, targets_list)
        metric_summary = self.metrics.compute()
        return metric_summary

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.DEVICE)
        outputs = self(images)

        if self.metrics:
            metric_summary = self._calculate_metrics(outputs, targets)
            metrics = {"val/map": metric_summary["map"]}
            self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.DEVICE)
        outputs = self(images)

        if self.metrics:
            metric_summary = self._calculate_metrics(outputs, targets)
            metrics = {"test/map": metric_summary["map"]}
            self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
