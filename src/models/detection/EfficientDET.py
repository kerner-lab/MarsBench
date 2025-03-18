from effdet import DetBenchTrain
from effdet import EfficientDet as EfficientDet_effdet
from effdet import get_efficientdet_config
from effdet.efficientdet import HeadNet

from .BaseDetectionModel import BaseDetectionModel


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
            print(
                "freeze_layers is set to True but model is not pretrained. Setting freeze_layers to False"
            )
            freeze_layers = False

        if pretrained and freeze_layers:
            for param in model.parameters():
                param.requires_grad = False

            for param in model.class_net.parameters():
                param.requires_grad = True
            for param in model.box_net.parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True

        return DetBenchTrain(model, config)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.DEVICE)

        loss_output_dict = self(images, targets)

        return loss_output_dict["loss"]

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.DEVICE)

        outputs = self(images, targets)

        loss = outputs["loss"]
        class_loss = outputs["class_loss"]
        box_loss = outputs["box_loss"]

        print(
            f"validation loss: {loss}, class_loss: {class_loss}, box_loss: {box_loss}"
        )

    def test_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.DEVICE)

        outputs = self(images, targets)

        loss = outputs["loss"]
        class_loss = outputs["class_loss"]
        box_loss = outputs["box_loss"]

        print(f"test loss: {loss}, class_loss: {class_loss}, box_loss: {box_loss}")
