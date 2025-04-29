"""
EfficientDET model implementation for object detection in Mars surface images.
"""

import logging

from effdet import DetBenchTrain
from effdet import EfficientDet as EfficientDet_effdet
from effdet import get_efficientdet_config
from effdet.efficientdet import HeadNet

from .BaseDetectionModel import BaseDetectionModel

logger = logging.getLogger(__name__)


class EfficientDET(BaseDetectionModel):
    def __init__(self, cfg):
        super(EfficientDET, self).__init__(cfg)

    def _initialize_model(self):
        num_classes = self.cfg.data.num_classes - 1
        architecture = self.cfg.model.architecture
        pretrained = self.cfg.model.pretrained
        freeze_layers = self.cfg.model.freeze_layers

        config = get_efficientdet_config(architecture)
        config.num_classes = num_classes
        config.image_size = tuple(self.cfg.model.input_size)[1:]

        model = EfficientDet_effdet(config, pretrained_backbone=pretrained)

        model.class_net = HeadNet(
            config,
            num_outputs=config.num_classes,
        )

        if freeze_layers and not pretrained:
            logger.warning("freeze_layers is set to True but model is not pretrained. Setting freeze_layers to False")
            freeze_layers = False

        if pretrained and freeze_layers:
            for param in model.parameters():
                param.requires_grad = False

            for param in model.class_net.parameters():
                param.requires_grad = True
            for param in model.box_net.parameters():
                param.requires_grad = True
            logger.info("Froze model parameters, keeping class and box network layers trainable")
        else:
            for param in model.parameters():
                param.requires_grad = True
            if pretrained:
                logger.info("Using pretrained weights with all layers trainable")
            else:
                logger.info("Training from scratch with all layers trainable")

        return DetBenchTrain(model, config)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_output_dict = self(images, targets)

        loss = loss_output_dict["loss"]

        self._log_metrics("train", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images, targets)

        # output keys: loss, class_loss, box_loss, detections
        loss = outputs["loss"]
        self._log_metrics("val", loss)

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images, targets)

        loss = outputs["loss"]
        self._log_metrics("test", loss)

        for i in range(len(images)):
            detections = outputs["detections"][i]
            self.test_outputs.append(
                {
                    "gt_bboxes": targets["bbox"][i].detach().cpu().numpy(),
                    "pred_bboxes": detections[:, :4].detach().cpu().numpy(),
                    "pred_score": detections[:, 4].detach().cpu().numpy(),
                }
            )
