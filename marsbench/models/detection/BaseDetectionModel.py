"""
Abstract base class for all Mars surface image detection models.
"""

from abc import ABC
from abc import abstractmethod

import pytorch_lightning as pl
import torch
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.sgd import SGD


class BaseDetectionModel(pl.LightningModule, ABC):
    def __init__(self, cfg):
        super(BaseDetectionModel, self).__init__()
        self.cfg = cfg
        self.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self._initialize_model().to(self.DEVICE)

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
        images = images.to(self.DEVICE)

        loss_dict = self(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        self._log_metrics("train", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.DEVICE)
        outputs = self(images)

        if self.metrics:
            metric_summary = self._calculate_metrics(outputs, targets)
            metrics = {"val/map": metric_summary["map"]}
            self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.DEVICE)
        outputs = self(images)

        if self.metrics:
            metric_summary = self._calculate_metrics(outputs, targets)
            metrics = {"test/map": metric_summary["map"]}
            self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)

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

        return optimizer

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
