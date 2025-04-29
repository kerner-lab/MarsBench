"""
Abstract base class for all Mars surface image detection models.
"""

from abc import ABC
from abc import abstractmethod

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.sgd import SGD

from marsbench.utils.detect_metrics import compute_object_metrics
from marsbench.utils.detect_metrics import match_bboxes


class BaseDetectionModel(pl.LightningModule, ABC):
    def __init__(self, cfg):
        super(BaseDetectionModel, self).__init__()
        self.cfg = cfg
        if self.cfg.training_type in ["scratch_training", "feature_extraction", "transfer_learning"]:
            self.cfg.model.pretrained = False if self.cfg.training_type == "scratch_training" else True
            self.cfg.model.freeze_layers = True if self.cfg.training_type == "feature_extraction" else False
        else:
            raise ValueError(f"Training type '{self.cfg.training_type}' not recognized.")
        self.model = self._initialize_model().to(self.device)
        self.test_outputs = []
        self.test_results = {}

        self.save_hyperparameters(cfg)

    @abstractmethod
    def _initialize_model(self):
        """Initialize the model architecture. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _initialize_model method.")

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def _log_metrics(self, prefix, loss, on_step=True, on_epoch=True):
        metrics = {f"{prefix}/loss": loss}
        self.log_dict(metrics, on_step=on_step, on_epoch=on_epoch, prog_bar=True)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.device)

        loss_dict = self(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        self._log_metrics("train", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.device)
        outputs = self(images)

        if self.metrics:
            metric_summary = self._calculate_metrics(outputs, targets)
            metrics = {"val/map": metric_summary["map"]}
            self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)

        self.model.train()
        loss_dict = self(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        self.model.eval()

        self._log_metrics("val", total_loss)

    def test_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.device)
        outputs = self(images)

        if self.metrics:
            metric_summary = self._calculate_metrics(outputs, targets)
            metrics = {"test/map": metric_summary["map"]}
            self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)

        for target, output in zip(targets, outputs):
            self.test_outputs.append(
                {
                    "gt_bboxes": target["boxes"].detach().cpu().numpy(),
                    "pred_bboxes": output["boxes"].detach().cpu().numpy(),
                    "pred_score": output["scores"].detach().cpu().numpy(),
                }
            )

    def on_test_epoch_end(self):
        object_iou = []
        object_accuracy = []
        object_precision = []
        object_recall = []

        for sample in self.test_outputs:
            object_iou.append(match_bboxes(sample["gt_bboxes"], sample["pred_bboxes"]))
            current_accuracy, current_precision, current_recall = compute_object_metrics(
                sample["gt_bboxes"], sample["pred_bboxes"], 0.5
            )
            object_accuracy.append(current_accuracy)
            object_precision.append(current_precision)
            object_recall.append(current_recall)

        object_iou_mean = np.mean(object_iou)
        object_accuracy_mean = np.mean(object_accuracy)
        object_precision_mean = np.mean(object_precision)
        object_recall_mean = np.mean(object_recall)

        self.test_results = {
            "object_iou_mean": round(float(object_iou_mean), 6),
            "object_accuracy_mean": round(float(object_accuracy_mean), 6),
            "object_precision_mean": round(float(object_precision_mean), 6),
            "object_recall_mean": round(float(object_recall_mean), 6),
        }

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

    def _calculate_metrics(self, outputs, targets):
        targets_list = []
        preds_list = []
        for output, target in zip(outputs, targets):
            targets_dict = {
                "boxes": target["boxes"].detach().cpu(),
                "labels": target["labels"].detach().cpu(),
            }
            preds_dict = {
                "boxes": output["boxes"].detach().cpu(),
                "labels": output["labels"].detach().cpu(),
                "scores": output["scores"].detach().cpu(),
            }
            targets_list.append(targets_dict)
            preds_list.append(preds_dict)

        self.metrics.reset()
        self.metrics.update(preds_list, targets_list)
        metric_summary = self.metrics.compute()
        return metric_summary
