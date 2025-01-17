import logging

from torch import nn
from torchvision.models import ViT_L_16_Weights
from torchvision.models import vit_l_16

from .BaseClassificationModel import BaseClassificationModel

logger = logging.getLogger(__name__)


class ViT(BaseClassificationModel):
    def __init__(self, cfg):
        super(ViT, self).__init__(cfg)

    def _initialize_model(self):
        num_classes = self.cfg.data.num_classes
        pretrained = self.cfg.model.classification.pretrained
        freeze_layers = self.cfg.model.classification.freeze_layers

        if pretrained:
            weights = ViT_L_16_Weights.DEFAULT
        else:
            weights = None

        model = vit_l_16(weights=weights)
        model.heads[-1] = nn.Linear(model.heads[-1].in_features, num_classes)

        if freeze_layers and not pretrained:
            logger.warning(
                "freeze_layers is set to True but model is not pretrained. Setting freeze_layers to False"
            )
            freeze_layers = False

        if pretrained and freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.heads[-1].parameters():
                param.requires_grad = True
            logger.info("Froze transformer layers, keeping final classifier trainable")
        else:
            for param in model.parameters():
                param.requires_grad = True
            logger.info("All layers are trainable")

        return model
