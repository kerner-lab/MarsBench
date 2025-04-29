"""
Mask2Former model implementation for Mars surface image segmentation.
"""

import logging

import matplotlib.pyplot as plt
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from torchvision.utils import make_grid
from transformers import Mask2FormerForUniversalSegmentation
from transformers import Mask2FormerImageProcessor

from .BaseSegmentationModel import BaseSegmentationModel

logger = logging.getLogger(__name__)


class Mask2Former(BaseSegmentationModel):
    def __init__(self, cfg):
        super(Mask2Former, self).__init__(cfg)

        # The image processor is initialized in the data module
        self.image_processor = Mask2FormerImageProcessor(
            ignore_index=self.cfg.training.ignore_index,
            reduce_labels=False,
        )

    def _initialize_model(self):

        pretrained = self.cfg.model.pretrained
        freeze_layers = self.cfg.model.freeze_layers

        # Set encoder weights based on pretrained flag
        # encoder_weights = "imagenet" if pretrained else None

        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-tiny-ade-semantic",
            num_labels=self.cfg.data.num_classes - 1,
            ignore_mismatched_sizes=True,
        )

        # Handle layer freezing
        if freeze_layers and not pretrained:
            logger.warning("freeze_layers is set to True but model is not pretrained. Setting freeze_layers to False")
            freeze_layers = False

        if pretrained and freeze_layers:
            # Freeze the entire pixel-level encoder
            for param in model.model.pixel_level_module.encoder.parameters():
                param.requires_grad = False
            # Keep decoder and transformer modules trainable for finetuning
            for param in model.model.pixel_level_module.decoder.parameters():
                param.requires_grad = True
            for param in model.model.transformer_module.parameters():
                param.requires_grad = True
            logger.info("Froze pixel-level encoder, keeping decoder and transformer trainable")
        else:
            # Make all layers trainable
            for param in model.model.parameters():
                param.requires_grad = True
            if pretrained:
                logger.info("Using pretrained weights with all layers trainable")
            else:
                logger.info("Training from scratch with all layers trainable")

        return model

    def forward(self, pixel_values, mask_labels, class_labels, pixel_mask):
        outputs = self.model(
            pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels, pixel_mask=pixel_mask
        )
        return outputs

    def _shared_step(self, batch, batch_idx, prefix):
        pixel_values = batch["pixel_values"].to(self.device)
        mask_labels = [mask_label.to(self.device) for mask_label in batch["mask_labels"]]
        class_labels = [class_label.to(self.device) for class_label in batch["class_labels"]]
        pixel_mask = batch["pixel_mask"].to(self.device)

        outputs = self(
            pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels, pixel_mask=pixel_mask
        )

        loss = outputs.loss

        # Get class id based output
        target_sizes = [(512, 512)] * len(batch["orig_mask"])
        pred_indices = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        pred_indices = torch.stack(pred_indices).to(self.device)
        target_masks = torch.stack(batch["orig_mask"], dim=0).to(self.device).long()

        metrics = {f"{prefix}/loss": loss}
        metrics.update(self._calculate_metrics_for_step(prefix, pred_indices, target_masks))
        # Log metrics at step level
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        if (
            self.log_images_every_n_epochs is not None
            and batch_idx == 0
            and self.trainer.is_global_zero
            and self.current_epoch % self.log_images_every_n_epochs == 0
        ):
            self._log_visualizations(batch["pixel_values"], batch["orig_mask"], pred_indices, prefix)

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        images = batch["pixel_values"].to(self.device)
        outputs = self(images)

        # Get class id based output
        target_sizes = [(512, 512)] * images.shape[0]
        pred_indices = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        pred_indices = torch.stack(pred_indices).to(self.device)

        return pred_indices

    def _log_visualizations(self, images, masks, preds, prefix):
        """
        Create and log visualization images.
        """

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
