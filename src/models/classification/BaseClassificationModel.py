from abc import ABC, abstractmethod
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.sgd import SGD
from torch.optim.adamw import AdamW
from torch.optim.adam import Adam


class BaseClassificationModel(pl.LightningModule, ABC):
    def __init__(self, cfg):
        super(BaseClassificationModel, self).__init__()
        self.cfg = cfg
        self.model = self._initialize_model()
        self.criterion = self._initialize_criterion()
        self.save_hyperparameters(cfg)

    @abstractmethod
    def _initialize_model(self):
        """Initialize the model architecture. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _initialize_model method.")

    def _initialize_criterion(self):
        criterion_name = self.cfg.criterion.name
        if criterion_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Criterion '{criterion_name}' not recognized.")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = self._calculate_accuracy(outputs, labels)

        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = self._calculate_accuracy(outputs, labels)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = self._calculate_accuracy(outputs, labels)

        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        optimizer_name = self.cfg.optimizer.name
        lr = self.cfg.optimizer.lr
        weight_decay = self.cfg.optimizer.get('weight_decay', 0.0)
        if optimizer_name == 'adam':
            optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = self.cfg.optimizer.get('momentum', 0.9)
            optimizer = SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer '{optimizer_name}' not recognized.")
        return optimizer


    def _calculate_accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        correct = (preds == labels).sum().float()
        acc = correct / labels.size(0)
        return acc
