from .BaseClassificationModel import BaseClassificationModel
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
from torch import nn
import warnings

class SqueezeNet(BaseClassificationModel):
    def __init__(self, cfg):
        super(SqueezeNet, self).__init__(cfg)

    def _initialize_model(self):
        num_classes = self.cfg.data.num_classes
        pretrained = self.cfg.model.classification.pretrained
        freeze_layers = self.cfg.model.classification.freeze_layers

        if pretrained:
            weights = SqueezeNet1_1_Weights.DEFAULT
        else:
            weights = None

        model = squeezenet1_1(weights=weights)
        # Replace the classifier to match num_classes
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes


        if freeze_layers and not pretrained:
            warnings.warn("freeze_layers is set to True but model is not pretrained. Setting freeze_layers to False")
            freeze_layers = False

        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier[1].parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True
        return model
