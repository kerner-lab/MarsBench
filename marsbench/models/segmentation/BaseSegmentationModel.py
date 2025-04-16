"""
Abstract base class for all Mars surface image segmentation models.
"""

from abc import ABC
from abc import abstractmethod
from typing import Literal

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.sgd import SGD
from torchmetrics import Accuracy
from torchmetrics import Precision
from torchmetrics import Recall
from torchmetrics.segmentation import GeneralizedDiceScore
from torchmetrics.segmentation import MeanIoU
from torchvision.utils import make_grid


class BaseSegmentationModel(pl.LightningModule, ABC):
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

        # Initialize metrics in setup() instead
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None
        self.test_results = {}
        self.visualization_samples = {}

        # Visualization settings
        self.log_images_every_n_epochs = self.cfg.training.get("log_images_every_n_epochs", 5)
        self.visualization_style = self.cfg.training.get("visualization_style", "grid")  # "grid" or "diff"
        self.overlay_alpha = self.cfg.training.get("overlay_alpha", 0.5)  # Transparency for visualizations
        self.max_samples = self.cfg.training.get("max_vis_samples", 4)  # Maximum number of samples to visualize

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

    def _create_metrics(self):
        """Create metrics for evaluation."""
        segmentation_config = {
            "num_classes": self.cfg.data.num_classes,
            "include_background": True,
            "input_format": "index",
        }
        classification_config = {"task": "multiclass", "num_classes": self.cfg.data.num_classes}

        metrics = torch.nn.ModuleDict(
            {
                "dice": GeneralizedDiceScore(
                    **segmentation_config, weight_type=self.cfg.training.criterion.get("weight_type", "square")
                ),
                "iou": MeanIoU(**segmentation_config),
                "accuracy": Accuracy(**classification_config),
                "precision": Precision(**classification_config, average="macro"),
                "recall": Recall(**classification_config, average="macro"),
                # Per-class metrics
                "per_class_dice": GeneralizedDiceScore(
                    **segmentation_config,
                    weight_type=self.cfg.training.criterion.get("weight_type", "square"),
                    per_class=True,
                ),
                "per_class_iou": MeanIoU(**segmentation_config, per_class=True),
                "per_class_precision": Precision(**classification_config, average=None),
                "per_class_recall": Recall(**classification_config, average=None),
            }
        )

        return metrics

    def setup(self, stage=None):
        """Called by PyTorch Lightning when the model is being set up."""
        # Create metrics for each stage
        self.train_metrics = self._create_metrics()
        self.val_metrics = self._create_metrics()
        self.test_metrics = self._create_metrics()

        # Initialize visualization samples
        self.visualization_samples = {
            "train": {"images": [], "masks": [], "outputs": []},
            "val": {"images": [], "masks": [], "outputs": []},
            "test": {"images": [], "masks": [], "outputs": []},
        }

    def on_fit_start(self):
        """Called by PyTorch Lightning when fit begins."""
        # Move metrics to the correct device
        device = self.device
        for metrics in [self.train_metrics, self.val_metrics, self.test_metrics]:
            for name, metric in metrics.items():
                metrics[name] = metric.to(device)

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

        # Get the correct metrics for the current phase
        metrics = getattr(self, f"{prefix}_metrics")

        # Update metrics
        for name, metric in metrics.items():
            if name.startswith("per_class_"):
                continue  # Handle per-class metrics separately
            metric.update(pred_indices, masks_long)

        # Update per-class metrics
        metrics["per_class_dice"].update(pred_indices, masks_long)
        metrics["per_class_iou"].update(pred_indices, masks_long)
        metrics["per_class_precision"].update(pred_indices, masks_long)
        metrics["per_class_recall"].update(pred_indices, masks_long)

        # Store visualization samples from first batch
        self._update_visualization_samples(prefix, images, masks, outputs, batch_idx)

        # Log loss for progress bar
        self.log(f"{prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)

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

    def _update_visualization_samples(self, prefix, images, masks, outputs, batch_idx):
        """Store visualization samples from first batch only."""
        if batch_idx == 0 and len(self.visualization_samples[prefix]["images"]) == 0:
            # Store only a few samples to save memory
            max_samples = min(4, images.shape[0])
            self.visualization_samples[prefix]["images"] = images[:max_samples].detach().cpu()
            self.visualization_samples[prefix]["masks"] = masks[:max_samples].detach().cpu()
            self.visualization_samples[prefix]["outputs"] = outputs[:max_samples].detach().cpu()

    def _create_diff_visualization(self, img, mask, pred, num_classes):
        """
        Create a difference visualization highlighting correct and incorrect predictions.

        Args:
            img: Original image tensor (C, H, W)
            mask: Ground truth mask tensor (H, W)
            pred: Prediction mask tensor (H, W)
            num_classes: Number of classes in the segmentation task

        Returns:
            Tuple of tensors (original_img, diff_map, pred_colored)
        """
        # Ensure image has 3 channels and is normalized
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        if img.max() > 1.0:
            img = img / 255.0

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

        # Create a colormap for class visualizations
        colormap = plt.cm.get_cmap("tab20", num_classes)

        # Create colored prediction mask
        pred_colored = torch.zeros((3, pred.shape[0], pred.shape[1]), dtype=torch.float32)

        for class_idx in range(num_classes):
            color = torch.tensor(colormap(class_idx)[:3], dtype=torch.float32)
            pred_locations = pred == class_idx

            for c in range(3):
                pred_colored[c][pred_locations] = color[c]

        return img, diff_overlay, pred_colored

    def _create_grid_visualization(self, img, mask, pred, num_classes):
        """
        Create a standard grid visualization with colored class masks.

        Args:
            img: Original image tensor (C, H, W)
            mask: Ground truth mask tensor (H, W)
            pred: Prediction mask tensor (H, W)
            num_classes: Number of classes in the segmentation task

        Returns:
            Tuple of tensors (original_img, colored_mask, colored_pred)
        """
        # Ensure image has 3 channels and is normalized
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        if img.max() > 1.0:
            img = img / 255.0

        # Create a colormap
        colormap = plt.cm.get_cmap("tab20", num_classes)

        # Convert masks to colored images using the colormap
        mask_vis = torch.zeros((3, mask.shape[0], mask.shape[1]), dtype=torch.float32)
        pred_vis = torch.zeros((3, pred.shape[0], pred.shape[1]), dtype=torch.float32)

        for class_idx in range(num_classes):
            # Get color for this class
            color = torch.tensor(colormap(class_idx)[:3], dtype=torch.float32)

            # Apply color to masks
            mask_locations = mask == class_idx
            pred_locations = pred == class_idx

            for c in range(3):  # RGB channels
                mask_vis[c][mask_locations] = color[c]
                pred_vis[c][pred_locations] = color[c]

        return img, mask_vis, pred_vis

    def _log_visualizations(self, prefix):
        """
        Create and log visualization images to Wandb.

        Creates visualizations based on the configured style:
        - "grid": Original image, ground truth mask, predicted mask
        - "diff": Original image, difference visualization, prediction mask
        """
        # Skip if no visualizations available or not using a logger
        if (
            not hasattr(self.logger, "experiment")
            or prefix not in self.visualization_samples
            or len(self.visualization_samples[prefix]["images"]) == 0
            or self.current_epoch % self.log_images_every_n_epochs != 0
        ):
            return

        images = self.visualization_samples[prefix]["images"]
        masks = self.visualization_samples[prefix]["masks"]
        outputs = self.visualization_samples[prefix]["outputs"]

        # Get predictions from outputs
        preds = torch.argmax(outputs, dim=1)

        num_samples = min(self.max_samples, len(images))
        num_classes = self.cfg.data.num_classes

        # Prepare visualization tensors
        all_images = []

        for i in range(num_samples):
            # Get image, ground truth, and prediction
            img = images[i].cpu()
            mask = masks[i].cpu()
            pred = preds[i].cpu()

            if self.visualization_style == "diff":
                # Create error visualization
                img, mask_vis, pred_vis = self._create_diff_visualization(img, mask, pred, num_classes)
            else:  # Default "grid" style
                img, mask_vis, pred_vis = self._create_grid_visualization(img, mask, pred, num_classes)

            # Add to the list of images
            all_images.extend([img, mask_vis, pred_vis])

        # Create a grid of images
        grid = make_grid(all_images, nrow=3)  # 3 columns: original, ground truth/diff, prediction

        # Log to wandb with appropriate caption
        if self.visualization_style == "diff":
            caption = f"Epoch {self.current_epoch}: Original | Diff (green=correct, red=error) | Prediction"
        else:
            caption = f"Epoch {self.current_epoch}: Original | Ground Truth | Prediction"

        self.logger.experiment.log(
            {f"{prefix}/visualizations": wandb.Image(grid, caption=caption), "epoch": self.current_epoch}
        )

    #######################
    # Lifecycle Hooks
    #######################

    def on_train_epoch_end(self):
        self._log_metrics_for_phase("train")
        self._log_visualizations("train")

    def on_validation_epoch_end(self):
        self._log_metrics_for_phase("val")
        self._log_visualizations("val")

    def on_test_epoch_end(self):
        self._log_metrics_for_phase("test")
        self._log_visualizations("test")

        # Ensure loss is also in test_results
        if "test/loss" in self.trainer.callback_metrics:
            self.test_results["loss"] = round(float(self.trainer.callback_metrics["test/loss"]), 4)

    def _log_metrics_for_phase(self, prefix):
        """Log all metrics for the given phase."""
        metrics = getattr(self, f"{prefix}_metrics")
        all_metrics = {}

        # Log regular metrics
        for name, metric in metrics.items():
            if not name.startswith("per_class_"):
                value = metric.compute()
                all_metrics[f"{prefix}/{name}"] = value

        # Log per-class metrics
        for metric_type in ["dice", "iou", "precision", "recall"]:
            per_class_values = metrics[f"per_class_{metric_type}"].compute()
            for class_idx in range(self.cfg.data.num_classes):
                class_name = self._get_class_name(class_idx)
                all_metrics[f"{prefix}/{class_name}_{metric_type}"] = per_class_values[class_idx]

                # Store in test_results for later use
                if prefix == "test":
                    self.test_results[f"{class_name}_{metric_type}"] = round(float(per_class_values[class_idx]), 4)

        self.log_dict(all_metrics, on_epoch=True, on_step=False, sync_dist=True)

        # Reset metrics after logging
        for name, metric in metrics.items():
            metric.reset()

        # Log current epoch to wandb
        if hasattr(self.logger, "experiment"):
            self.logger.experiment.log({"epoch": self.current_epoch})
