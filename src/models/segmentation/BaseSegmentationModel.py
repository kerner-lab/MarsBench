from abc import ABC
from abc import abstractmethod
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.sgd import SGD


class BaseSegmentationModel(pl.LightningModule, ABC):
    """Abstract base class for segmentation models."""

    def __init__(self, cfg):
        super(BaseSegmentationModel, self).__init__()
        self.cfg = cfg
        self.model = self._initialize_model()
        self.criterion = self._initialize_criterion()
        self.save_hyperparameters(cfg)

    def _get_in_channels(self) -> int:
        """Get number of input channels based on image type."""
        image_type = self.cfg.data.image_type.lower().strip()
        if image_type in ["rgb", "bgr"]:
            return 3
        elif image_type in ["grayscale", "l"]:
            return 1
        else:
            raise ValueError(
                f"Unsupported image type: {image_type}. Must be one of: rgb, bgr, grayscale, l"
            )

    @abstractmethod
    def _initialize_model(self):
        """Initialize the model architecture. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _initialize_model method.")

    def _initialize_criterion(self):
        criterion_name = self.cfg.criterion.name
        if criterion_name == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif criterion_name == "dice":
            return self.dice_loss
        elif criterion_name == "combined":
            # Combine CrossEntropy and Dice loss
            return lambda pred, target: (
                nn.CrossEntropyLoss()(pred, target) + self.dice_loss(pred, target)
            )
        else:
            raise ValueError(f"Criterion '{criterion_name}' not recognized.")

    def dice_loss(self, pred, target, smooth=1.0):
        pred = F.softmax(pred, dim=1)
        pred = pred.contiguous()
        target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)
        target = target.contiguous()

        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        metrics = self._calculate_metrics(outputs, masks)

        # Log all metrics
        self.log("train_loss", loss)
        for metric_name, value in metrics.items():
            self.log(f"train_{metric_name}", value, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        metrics = self._calculate_metrics(outputs, masks)

        # Log all metrics
        self.log("val_loss", loss, prog_bar=True)
        for metric_name, value in metrics.items():
            self.log(f"val_{metric_name}", value, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        metrics = self._calculate_metrics(outputs, masks)

        # Log all metrics
        self.log("test_loss", loss)
        for metric_name, value in metrics.items():
            self.log(f"test_{metric_name}", value)

    def configure_optimizers(self):
        optimizer_name = self.cfg.optimizer.name
        lr = self.cfg.optimizer.lr
        weight_decay = self.cfg.optimizer.get("weight_decay", 0.0)

        if optimizer_name == "adam":
            optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            momentum = self.cfg.optimizer.get("momentum", 0.9)
            optimizer = SGD(
                self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Optimizer '{optimizer_name}' not recognized.")

        return optimizer

    def _calculate_metrics(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate segmentation metrics.

        Args:
            outputs: Model predictions [B, C, H, W]
            targets: Ground truth masks [B, H, W]

        Returns:
            Dictionary containing metrics (IoU, Dice coefficient)
        """
        # Get predicted class indices
        pred_masks = torch.argmax(outputs, dim=1)

        # Calculate IoU
        intersection = torch.logical_and(pred_masks, targets)
        union = torch.logical_or(pred_masks, targets)
        iou = (intersection.sum().float() + 1e-8) / (union.sum().float() + 1e-8)

        # Calculate Dice coefficient
        dice = (2 * intersection.sum().float() + 1e-8) / (
            pred_masks.sum().float() + targets.sum().float() + 1e-8
        )

        # Calculate pixel accuracy
        correct = (pred_masks == targets).sum().float()
        total = targets.numel()
        accuracy = correct / total

        return {"iou": iou, "dice": dice, "accuracy": accuracy}
