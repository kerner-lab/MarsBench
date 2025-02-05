from abc import abstractmethod

import pytorch_lightning as pl
import torch
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.sgd import SGD


class BaseDetectionModel(pl.LightningModule):
    def __init__(self, cfg):
        super(BaseDetectionModel, self).__init__()
        self.cfg = cfg
        self.DEVICE = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = self._initialize_model().to(self.DEVICE)

    @abstractmethod
    def _initialize_model(self):
        """Initialize the model architecture. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _initialize_model method.")

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.DEVICE)

        loss_dict = self(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        return total_loss

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
