from effdet import DetBenchTrain
from effdet import EfficientDet as EfficientDet_effdet
from effdet import get_efficientdet_config
from effdet.config.model_config import efficientdet_model_param_dict
from effdet.efficientdet import HeadNet

from .BaseDetectionModel import BaseDetectionModel


class EfficientDET(BaseDetectionModel):
    def __init__(self, cfg):
        super(EfficientDET, self).__init__(cfg)
        self.img_size = self.cfg.data.image_size
        self.prediction_confidence_threshold = (
            self.cfg.model.detection.prediction_confidence_threshold
        )
        self.lr = self.cfg.optimizer.lr
        self.wbf_iou_threshold = self.cfg.model.detection.wbf_iou_threshold

    def _initialize_model(self):
        num_classes = self.cfg.data.num_classes
        architecture = self.cfg.model.detection.architecture
        image_size = self.cfg.data.image_size
        pretrained = self.cfg.model.detection.pretrained
        freeze_layers = self.cfg.model.detection.freeze_layers

        efficientdet_model_param_dict["tf_efficientnetv2_l"] = dict(
            name="tf_efficientnetv2_l",
            backbone_name="tf_efficientnetv2_l",
            backbone_args=dict(drop_path_rate=0.2),
            num_classes=num_classes,
            url="",
        )

        config = get_efficientdet_config(architecture)
        config.update({"num_classes": num_classes})
        config.update({"image_size": (image_size, image_size)})

        # config.head_bn_level_first = True  # Better for TF-style models
        # decides orders of BN and conv in head

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

        if freeze_layers:
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
