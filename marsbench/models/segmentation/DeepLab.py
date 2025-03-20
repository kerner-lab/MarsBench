import logging

import segmentation_models_pytorch as smp

from .BaseSegmentationModel import BaseSegmentationModel

logger = logging.getLogger(__name__)


class DeepLab(BaseSegmentationModel):
    def __init__(self, cfg):
        super(DeepLab, self).__init__(cfg)

    def _initialize_model(self):
        """Initialize DeepLab model with configuration parameters."""
        in_channels = self._get_in_channels()
        num_classes = self.cfg.data.num_classes
        encoder_name = self.cfg.model.get("encoder_name", "resnet34")
        pretrained = self.cfg.model.pretrained
        freeze_layers = self.cfg.model.freeze_layers

        # Set encoder weights based on pretrained flag
        encoder_weights = "imagenet" if pretrained else None

        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

        # Handle layer freezing
        if freeze_layers and not pretrained:
            logger.warning("freeze_layers is set to True but model is not pretrained. Setting freeze_layers to False")
            freeze_layers = False

        if pretrained and freeze_layers:
            # Freeze encoder layers
            for param in model.encoder.parameters():
                param.requires_grad = False
            # Keep decoder trainable for fine-tuning
            for param in model.decoder.parameters():
                param.requires_grad = True
            for param in model.segmentation_head.parameters():
                param.requires_grad = True
            logger.info("Froze encoder layers, keeping decoder and segmentation head trainable")
        else:
            # Make all layers trainable
            for param in model.parameters():
                param.requires_grad = True
            if pretrained:
                logger.info("Using pretrained weights with all layers trainable")
            else:
                logger.info("Training from scratch with all layers trainable")

        return model
