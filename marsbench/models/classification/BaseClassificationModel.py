"""
Abstract base class for all Mars surface image classification models.
"""

from abc import ABC
from abc import abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.sgd import SGD
from torchmetrics import AUROC
from torchmetrics import Accuracy
from torchmetrics import F1Score
from torchmetrics import Precision
from torchmetrics import Recall


class BaseClassificationModel(pl.LightningModule, ABC):
    """Abstract base class for classification models."""

    def __init__(self, cfg):
        super(BaseClassificationModel, self).__init__()
        self.cfg = cfg
        if self.cfg.training_type in ["scratch_training", "feature_extraction", "transfer_learning"]:
            self.cfg.model.pretrained = False if self.cfg.training_type == "scratch_training" else True
            self.cfg.model.freeze_layers = True if self.cfg.training_type == "feature_extraction" else False
        else:
            raise ValueError(f"Training type '{self.cfg.training_type}' not recognized.")
        self.model = self._initialize_model()
        self.criterion = self._initialize_criterion()
        self.save_hyperparameters(cfg)
        self.val_outputs = []
        self.test_outputs = []

        # metrics
        self.test_results = {}
        self.accuracy = Accuracy(task="multiclass", num_classes=self.cfg.data.num_classes)
        self.precision = Precision(task="multiclass", num_classes=self.cfg.data.num_classes)
        self.recall = Recall(task="multiclass", num_classes=self.cfg.data.num_classes)
        self.f1 = F1Score(
            task="multiclass", num_classes=self.cfg.data.num_classes, multidim_average="global", average="weighted"
        )
        self.auroc = AUROC(task="multiclass", num_classes=self.cfg.data.num_classes)

    @abstractmethod
    def _initialize_model(self):
        """Initialize the model architecture. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _initialize_model method.")

    def _initialize_criterion(self):
        criterion_name = self.cfg.training.criterion.name
        if criterion_name == "cross_entropy":
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Criterion '{criterion_name}' not recognized.")

    def forward(self, x):
        return self.model(x)

    def _log_metrics(self, prefix, loss, acc, on_step=True, on_epoch=True):
        """Helper method to log metrics consistently."""
        self.log(f"{prefix}/loss", loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True)
        self.log(f"{prefix}/acc", acc, on_step=on_step, on_epoch=on_epoch, prog_bar=True)

    def _log_predictions(self, batch_idx, images, labels, outputs, prefix="train", max_samples=4):
        """Helper method to log predictions periodically."""
        if not hasattr(self.logger, "experiment"):
            return

        if batch_idx % 100 == 0:  # Log every 100 batches
            _, preds = torch.max(outputs, 1)
            num_samples = min(max_samples, len(images))

            for idx in range(num_samples):
                self.logger.experiment.log(
                    {
                        f"{prefix}_predictions": wandb.Image(
                            images[idx],
                            caption=f"True: {labels[idx].item()}, Pred: {preds[idx].item()}",
                        )
                    }
                )

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = self._calculate_accuracy(outputs, labels)

        # Log metrics
        self._log_metrics("train", loss, acc)
        self._log_predictions(batch_idx, images, labels, outputs)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = self._calculate_accuracy(outputs, labels)

        # Log metrics
        self._log_metrics("val", loss, acc, on_step=False)

        # Store outputs for epoch end
        self.val_outputs.append({"preds": outputs.detach(), "labels": labels.detach()})

        return loss

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return

        # Clear stored outputs
        self.val_outputs.clear()

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = self._calculate_accuracy(outputs, labels)

        # Log metrics
        self._log_metrics("test", loss, acc, on_step=False)

        # Store predictions for confusion matrix
        self.test_outputs.append({"preds": outputs.detach(), "labels": labels.detach()})

        return loss

    def on_test_epoch_end(self):
        if not self.test_outputs or (self.logger and not hasattr(self.logger, "experiment")):
            return

        # Aggregate predictions
        all_preds = torch.cat([x["preds"] for x in self.test_outputs])
        all_labels = torch.cat([x["labels"] for x in self.test_outputs])

        # Get predicted classes
        _, preds = torch.max(all_preds, dim=1)

        # Log confusion matrix
        if self.logger:
            self.logger.experiment.log(
                {
                    "test/confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=all_labels.cpu().numpy(),
                        preds=preds.cpu().numpy(),
                    )
                }
            )

        self.test_results = self._calculate_metrics_benchmark(all_preds, all_labels)

        # Clear stored outputs
        self.test_outputs.clear()

    def predict_step(self, batch, batch_idx):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        outputs = self(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        return {"probabilities": probabilities, "predictions": predictions}

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

    def _calculate_accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        correct = (preds == labels).sum().float()
        acc = correct / labels.size(0)
        return acc

    def _calculate_metrics_benchmark(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)

        accuracy = self.accuracy(preds, labels)
        precision = self.precision(preds, labels)
        recall = self.recall(preds, labels)
        f1_score = self.f1(preds, labels)
        auroc_score = self.auroc(probs, labels)

        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.auroc.reset()

        return {
            "accuracy": round(float(accuracy.item()), 4),
            "precision": round(float(precision.item()), 4),
            "recall": round(float(recall.item()), 4),
            "f1_score": round(float(f1_score.item()), 4),
            "auroc_score": round(float(auroc_score.item()), 4),
        }
