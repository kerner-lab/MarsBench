"""
Abstract base class for all Mars surface image segmentation models.
"""

import logging
from abc import ABC
from abc import abstractmethod
from typing import Literal

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.sgd import SGD
from torchmetrics.functional import accuracy
from torchmetrics.functional import precision
from torchmetrics.functional import recall
from torchmetrics.functional.segmentation import generalized_dice_score
from torchmetrics.functional.segmentation import mean_iou
from torchvision.utils import make_grid

logger = logging.getLogger(__name__)


class BaseSegmentationModel(LightningModule, ABC):
    """Abstract base class for segmentation models."""

    #######################
    # Initialization & Setup
    #######################

    def __init__(self, cfg):
        super(BaseSegmentationModel, self).__init__()
        self.cfg = cfg
        self.ignore_index = self.cfg.training.get("ignore_index", -100)

        if self.cfg.training_type in ["scratch_training", "feature_extraction", "transfer_learning"]:
            self.cfg.model.pretrained = False if self.cfg.training_type == "scratch_training" else True
            self.cfg.model.freeze_layers = True if self.cfg.training_type == "feature_extraction" else False
        else:
            raise ValueError(f"Training type '{self.cfg.training_type}' not recognized.")
        self.model = self._initialize_model()
        self.criterion = self._initialize_criterion()

        # Test results
        self.test_results = {}

        # Visualization settings
        self.log_images_every_n_epochs = self.cfg.logger.get("log_images_every_n_epochs", 3)
        self.overlay_alpha = self.cfg.logger.get("overlay_alpha", 0.5)  # Transparency for visualizations
        self.max_samples = self.cfg.logger.get("max_vis_samples", 4)  # Maximum number of samples to visualize

        self.save_hyperparameters(cfg)

    #######################
    # Initialization & Setup
    #######################

    def _get_in_channels(self) -> int:
        """Get number of input channels based on image type."""
        image_type = self.cfg.data.image_type.lower().strip()
        if image_type in ["rgb", "bgr"]:
            return 3
        elif image_type in ["grayscale", "l"]:
            return 1
        else:
            raise ValueError(f"Unsupported image type: {image_type}. Must be one of: rgb, bgr, grayscale, l")

    @abstractmethod
    def _initialize_model(self):
        """Initialize the model architecture. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _initialize_model method.")

    def _initialize_criterion(self):
        criterion_name = self.cfg.training.criterion.name.lower()

        if criterion_name == "cross_entropy":
            return nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        elif criterion_name == "dice":
            return self.dice_loss
        elif criterion_name == "generalized_dice":
            weight_type = self.cfg.training.criterion.get("weight_type", "square")
            return lambda pred, target: self.generalized_dice_loss(pred, target, weight_type=weight_type)
        else:
            raise ValueError(f"Criterion '{criterion_name}' not recognized.")

    #######################
    # Model Infrastructure
    #######################

    def configure_optimizers(self):
        optimizer_name = self.cfg.training.optimizer.name
        lr = self.cfg.training.optimizer.lr
        weight_decay = self.cfg.training.optimizer.get("weight_decay", 0.0)

        if optimizer_name.lower() == "adam":
            optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == "adamw":
            optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == "sgd":
            momentum = self.cfg.training.optimizer.get("momentum", 0.9)
            optimizer = SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer '{optimizer_name}' not recognized.")

        if not self.cfg.training.get("scheduler", {}).get("enabled", False):
            return optimizer

        scheduler_name = self.cfg.training.scheduler.name
        scheduler_params = self.cfg.training.scheduler

        if scheduler_name.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_params.get("t_max", self.cfg.training.trainer.max_epochs),
                eta_min=scheduler_params.get("eta_min", 0),
            )
        elif scheduler_name.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_params.get("step_size", 10),
                gamma=scheduler_params.get("gamma", 0.1),
            )
        elif scheduler_name.lower() == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=scheduler_params.get("patience", 5),
                factor=scheduler_params.get("factor", 0.1),
                mode=scheduler_params.get("mode", "min"),
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}
        else:
            raise ValueError(f"Scheduler '{scheduler_name}' not recognized.")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x):
        return self.model(x)

    #######################
    # Loss Functions
    #######################

    def dice_loss(self, pred, target, smooth: float = 1.0):
        pred = F.softmax(pred, dim=1)

        # One hot encode target
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # Calculate intersection and union
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        # Calculate dice coefficient and loss
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()

        return dice_loss

    def generalized_dice_loss(
        self, pred, target, weight_type: Literal["uniform", "simple", "square"] = "square", smooth: float = 1e-5
    ):
        """
        Compute the Generalized Dice Loss.

        GDS = [2 * Σ_c (w_c * Σ_i (p_ci * t_ci)) + smooth] / [Σ_c (w_c * Σ_i (p_ci + t_ci)) + smooth]

        where:
            - p_ci: the predicted probability (from softmax) for class c at pixel i,
            - t_ci: the target probability (from one-hot encoding) for class c at pixel i,
            - w_c: the weight for class c, which depends on `weight_type`:
                - "uniform": all weights are 1
                - "simple": w_c = 1 / Σ_i t_ci
                - "square": w_c = 1 / (Σ_i t_ci)²
            - smooth: a small constant to avoid division by zero.
        """
        # Apply softmax to convert logits to probabilities
        pred_probs = F.softmax(pred, dim=1)

        # One-hot encode the target
        target = target.long()
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # Calculate the class weights based on weight_type
        if weight_type == "uniform":
            weights = torch.ones(pred.shape[1], device=pred.device)
        else:
            # Calculate the sum of target probabilities for each class
            target_sums = target_one_hot.sum(dim=(0, 2, 3))

            # Avoid division by zero
            target_sums = torch.clamp(target_sums, min=smooth)

            if weight_type == "simple":
                weights = 1.0 / target_sums
            elif weight_type == "square":
                weights = 1.0 / (target_sums**2)
            else:
                raise ValueError(f"Unknown weight_type: {weight_type}")

        # Calculate intersection and union
        intersection = torch.sum(pred_probs * target_one_hot, dim=(0, 2, 3))
        union = torch.sum(pred_probs + target_one_hot, dim=(0, 2, 3))

        # Apply weights
        weighted_intersection = weights * intersection
        weighted_union = weights * union

        # Calculate the generalized Dice score
        dice = (2.0 * torch.sum(weighted_intersection) + smooth) / (torch.sum(weighted_union) + smooth)

        # Convert to loss
        loss = 1.0 - dice

        return loss

    #######################
    # Step Functions
    #######################

    def _shared_step(self, batch, batch_idx, prefix):
        """Shared logic for training, validation and test steps."""
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)

        pred_indices = torch.argmax(outputs, dim=1).long()
        masks_long = masks.long()
        metrics = {f"{prefix}/loss": loss}
        metrics.update(self._calculate_metrics_for_step(prefix, pred_indices, masks_long))
        # Log metrics at step level
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        if (
            self.log_images_every_n_epochs is not None
            and batch_idx == 0
            and self.trainer.is_global_zero
            and self.current_epoch % self.log_images_every_n_epochs == 0
        ):
            self._log_visualizations(images, masks, outputs, prefix)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        outputs = self(images)

        pred_masks = torch.argmax(outputs, dim=1) if outputs.shape[1] > 1 else (outputs > 0.5).float()

        return pred_masks

    #######################
    # Metrics & Visualization
    #######################

    def _get_class_name(self, class_idx):
        """Get human-readable class name for given class index."""
        return (
            f"class_{class_idx}"
            if not self.cfg.get("mapping")
            else self.cfg.mapping.get(class_idx, f"class_{class_idx}")
        )

    def _calculate_metrics_for_step(self, prefix, pred_indices, masks_long) -> dict:
        """Calculate metrics for a single step."""
        metrics = {}
        # Compute functional metrics per batch
        num_classes = self.cfg.data.num_classes
        metrics[f"{prefix}/accuracy"] = accuracy(pred_indices, masks_long, task="multiclass", num_classes=num_classes)
        metrics[f"{prefix}/precision"] = precision(pred_indices, masks_long, task="multiclass", num_classes=num_classes)
        metrics[f"{prefix}/recall"] = recall(pred_indices, masks_long, task="multiclass", num_classes=num_classes)
        metrics[f"{prefix}/dice"] = generalized_dice_score(
            pred_indices, masks_long, input_format="index", num_classes=num_classes
        ).mean()
        metrics[f"{prefix}/iou"] = mean_iou(
            pred_indices, masks_long, input_format="index", num_classes=num_classes
        ).mean()

        # Compute per-class metrics
        per_class_accuracy = accuracy(
            pred_indices, masks_long, task="multiclass", num_classes=num_classes, average="none"
        )
        per_class_precision = precision(
            pred_indices, masks_long, task="multiclass", num_classes=num_classes, average="none"
        )
        per_class_recall = recall(pred_indices, masks_long, task="multiclass", num_classes=num_classes, average="none")
        per_class_dice = generalized_dice_score(
            pred_indices, masks_long, input_format="index", num_classes=num_classes, per_class=True
        ).mean(dim=0)
        per_class_iou = mean_iou(
            pred_indices, masks_long, input_format="index", num_classes=num_classes, per_class=True
        ).mean(dim=0)
        for i, (acc, prec, rec, dice, iou) in enumerate(
            zip(per_class_accuracy, per_class_precision, per_class_recall, per_class_dice, per_class_iou)
        ):
            class_name = self._get_class_name(i)
            metrics[f"{prefix}/{class_name}/accuracy"] = acc.item()
            metrics[f"{prefix}/{class_name}/precision"] = prec.item()
            metrics[f"{prefix}/{class_name}/recall"] = rec.item()
            metrics[f"{prefix}/{class_name}/dice"] = dice.item()
            metrics[f"{prefix}/{class_name}/iou"] = iou.item()
        return metrics

    def _log_visualizations(self, images, masks, outputs, prefix):
        """
        Create and log visualization images.
        """
        # Get predictions from outputs
        preds = torch.argmax(outputs, dim=1)

        num_samples = min(self.max_samples, len(images))
        colormap = plt.cm.get_cmap("tab20", self.cfg.data.num_classes)

        # Prepare visualization tensors
        all_images = []
        for i in range(num_samples):
            # Get image, ground truth, and prediction
            img = images[i].cpu()
            mask = masks[i].cpu()
            pred = preds[i].cpu()

            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)

            m = torch.tensor(self.cfg.transforms.rgb.mean).view(3, 1, 1)
            s = torch.tensor(self.cfg.transforms.rgb.std).view(3, 1, 1)
            img = img * s + m
            img = img.clamp(0, 1)

            # Create a difference mask (red=wrong, green=correct)
            diff_mask = torch.zeros((3, mask.shape[0], mask.shape[1]), dtype=torch.float32)

            # Mark correct predictions in green
            correct = mask == pred
            diff_mask[1, correct] = 1.0  # Green channel

            # Mark incorrect predictions in red
            incorrect = mask != pred
            diff_mask[0, incorrect] = 1.0  # Red channel

            # Overlay diff on the original image
            alpha = self.overlay_alpha
            diff_overlay = img * (1 - alpha) + diff_mask * alpha

            # Create a difference visualization
            mask_vis = torch.zeros((3, mask.shape[0], mask.shape[1]), dtype=torch.float32)
            pred_vis = torch.zeros((3, pred.shape[0], pred.shape[1]), dtype=torch.float32)

            for class_idx in range(self.cfg.data.num_classes):
                color = torch.tensor(colormap(class_idx)[:3], dtype=torch.float32)
                mask_locations = mask == class_idx
                pred_locations = pred == class_idx

                for c in range(3):  # RGB channels
                    mask_vis[c][mask_locations] = color[c]
                    pred_vis[c][pred_locations] = color[c]

            # Add to the list of images
            all_images.extend([img, mask_vis, pred_vis, diff_overlay])

        # Create a grid of images and log via Lightning
        grid = make_grid(all_images, nrow=4)
        caption = [
            f"Epoch {self.current_epoch}: Original | Ground Truth | Prediction | Diff (green=correct, red=error)"
        ]
        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(
                f"visualizations/{prefix}",
                [grid],
                caption=caption,
                step=self.current_epoch,
            )
        elif isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image(
                f"visualizations/{prefix}",
                grid,
                global_step=self.current_epoch,
            )
        else:
            logger.warning(f"Logger {self.logger} does not support image logging.")

    #######################
    # Lifecycle Hooks
    #######################
    def on_test_epoch_end(self):
        for metric in self.trainer.callback_metrics:
            if metric.startswith("test/"):
                self.test_results[metric] = round(float(self.trainer.callback_metrics[metric]), 4)
