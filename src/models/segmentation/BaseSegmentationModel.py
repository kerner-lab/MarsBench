from abc import ABC
from abc import abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
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
        self.test_outputs = []

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

    def _log_metrics(self, prefix, loss, dice, iou, on_step=True, on_epoch=True):
        """Helper method to log metrics consistently."""
        metrics = {f"{prefix}/loss": loss, f"{prefix}/dice": dice, f"{prefix}/iou": iou}
        self.log_dict(metrics, on_step=on_step, on_epoch=on_epoch, prog_bar=True)

    def _log_segmentation(
        self, batch_idx, images, masks, outputs, prefix="train", max_samples=4
    ):
        """Helper method to log segmentation visualizations."""
        if not hasattr(self.logger, "experiment") or batch_idx % 100 != 0:
            return

        # Get predicted masks
        pred_masks = (
            torch.argmax(outputs, dim=1)
            if outputs.shape[1] > 1
            else (outputs > 0.5).float()
        )

        # Log sample predictions
        num_samples = min(max_samples, len(images))
        for idx in range(num_samples):
            self.logger.experiment.log(
                {
                    f"{prefix}_segmentation": wandb.Image(
                        images[idx].cpu(),
                        masks={
                            "predictions": {
                                "mask_data": pred_masks[idx].cpu().numpy(),
                            },
                            "ground_truth": {
                                "mask_data": masks[idx].cpu().numpy(),
                            },
                        },
                    )
                }
            )

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)

        # Calculate metrics
        dice = self._calculate_dice_score(outputs, masks)
        iou = self._calculate_iou_score(outputs, masks)

        # Log metrics and visualizations
        self._log_metrics("train", loss, dice, iou)
        self._log_segmentation(batch_idx, images, masks, outputs)

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)

        # Calculate metrics
        dice = self._calculate_dice_score(outputs, masks)
        iou = self._calculate_iou_score(outputs, masks)

        # Log metrics
        self._log_metrics("val", loss, dice, iou, on_step=False)

        return loss

    def test_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)

        # Calculate metrics
        dice = self._calculate_dice_score(outputs, masks)
        iou = self._calculate_iou_score(outputs, masks)

        # Log metrics
        self._log_metrics("test", loss, dice, iou, on_step=False)

        # Store outputs for per-class analysis
        self.test_outputs.append({"outputs": outputs.detach(), "masks": masks.detach()})

        return loss

    def predict_step(self, batch, batch_idx):
        """Prediction step for segmentation model.

        Args:
            batch: Input batch, can be just images or tuple of (images, masks)
            batch_idx: Index of the current batch

        Returns:
            Predicted segmentation masks
        """
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        outputs = self(images)

        # Convert logits to predicted masks
        pred_masks = (
            torch.argmax(outputs, dim=1)
            if outputs.shape[1] > 1
            else (outputs > 0.5).float()
        )

        return pred_masks

    def on_test_epoch_end(self):
        """Calculate and log per-class metrics at the end of test epoch."""
        if not self.test_outputs or not hasattr(self.logger, "experiment"):
            return

        # Aggregate all outputs
        all_outputs = torch.cat([x["outputs"] for x in self.test_outputs])
        all_masks = torch.cat([x["masks"] for x in self.test_outputs])

        # Calculate per-class metrics
        num_classes = all_outputs.shape[1]
        for class_idx in range(num_classes):
            class_pred = all_outputs[:, class_idx]
            class_true = all_masks == class_idx

            dice = self._calculate_dice_score(class_pred, class_true)
            iou = self._calculate_iou_score(class_pred, class_true)

            self.logger.experiment.log(
                {
                    f"test/class_{class_idx}_dice": dice,
                    f"test/class_{class_idx}_iou": iou,
                }
            )

        # Clear stored outputs
        self.test_outputs.clear()

    def _calculate_dice_score(self, outputs, targets):
        """Calculate Dice score between predicted and target masks.

        Dice = 2|Intersection| / (|A| + |B|)
        where |A| and |B| are the cardinalities of the two sets.
        """
        if outputs.shape != targets.shape:
            outputs = (outputs > 0.5).float()

        intersection = (outputs * targets).sum()
        total_elements = outputs.sum() + targets.sum()  # |A| + |B|

        return (2.0 * intersection + 1e-8) / (total_elements + 1e-8)

    def _calculate_iou_score(self, outputs, targets):
        """Calculate IoU (Intersection over Union) score between predicted and target masks.

        IoU = |Intersection| / |Union| = |Intersection| / (|A| + |B| - |Intersection|)
        where |A| and |B| are the cardinalities of the two sets.
        """
        if outputs.shape != targets.shape:
            outputs = (outputs > 0.5).float()

        intersection = (outputs * targets).sum()
        set_union = (
            outputs + targets
        ).sum() - intersection  # |A| + |B| - |Intersection|

        return (intersection + 1e-8) / (set_union + 1e-8)

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
