"""
Abstract base class for all Mars surface image segmentation models.
"""

import logging
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from torch.optim import SGD
from torch.optim import Adam
from torch.optim import AdamW
from torchmetrics import Accuracy
from torchmetrics import MetricCollection
from torchmetrics import Precision
from torchmetrics import Recall
from torchmetrics.segmentation import GeneralizedDiceScore
from torchmetrics.segmentation import MeanIoU
from torchvision.utils import make_grid

from marsbench.utils.load_mapping import get_class_name

logger = logging.getLogger(__name__)


class BaseSegmentationModel(LightningModule, ABC):
    """Abstract base class for segmentation models."""

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ignore_index = self.cfg.training.get("ignore_index", -100)
        if self.cfg.training_type in ["scratch_training", "feature_extraction", "transfer_learning"]:
            self.cfg.model.pretrained = False if self.cfg.training_type == "scratch_training" else True
            self.cfg.model.freeze_layers = True if self.cfg.training_type == "feature_extraction" else False
        else:
            raise ValueError(f"Training type '{self.cfg.training_type}' not recognized.")

        # backbone / head built by subclass
        self.model = self._initialize_model()
        self.criterion = self._initialize_criterion()

        # MetricCollection (single base cloned per phase) ---
        C = cfg.data.num_classes
        weight_type = cfg.training.criterion.get("weight_type", "square")
        base = MetricCollection(
            {
                "acc": Accuracy(task="multiclass", num_classes=C, average=None),
                "prec": Precision(task="multiclass", num_classes=C, average=None),
                "rec": Recall(task="multiclass", num_classes=C, average=None),
                "dice": GeneralizedDiceScore(
                    num_classes=C, weight_type=weight_type, per_class=True, input_format="index"
                ),
                "iou": MeanIoU(num_classes=C, per_class=True, input_format="index"),
            }
        )
        self.train_metrics = base.clone(prefix="train/")
        self.val_metrics = base.clone(prefix="val/")
        self.test_metrics = base.clone(prefix="test/")

        # visual logging
        self.vis_every = cfg.logger.get("vis_every", 3)
        self.overlay_alpha = cfg.logger.get("overlay_alpha", 0.5)
        self.max_vis = cfg.logger.get("max_vis_samples", 4)
        # samples stored as {phase: (imgs, gt, preds)}
        self.vis_samples: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        cmap = plt.get_cmap("tab20", C)
        self.color_lut = torch.tensor([cmap(i)[:3] for i in range(C)])  # [C,3]

        self._legend_logged = False
        self.save_hyperparameters(cfg)

        self.test_results = {}

    def _get_in_channels(self) -> int:
        """Determine number of input channels based on configuration."""
        input_size = getattr(self.cfg.model, "input_size", None)
        if input_size:
            return int(input_size[0])
        image_type = self.cfg.data.image_type.lower()
        if image_type in ("rgb",):
            return 3
        elif image_type in ("grayscale", "l"):
            return 1
        else:
            logger.warning(f"Unknown image_type '{self.cfg.data.image_type}', defaulting to 3 channels")
            return 3

    # ---------------- abstract hooks ----------------
    @abstractmethod
    def _initialize_model(self) -> nn.Module:
        raise NotImplementedError

    # ---------------- criterion ---------------------
    def _initialize_criterion(self):
        name = self.cfg.training.criterion.name.lower()
        if name == "cross_entropy":
            return nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        if name == "dice":
            return self._dice_loss
        if name == "generalized_dice":
            wt = self.cfg.training.criterion.get("weight_type", "square")
            return lambda p, t: self._generalized_dice_loss(p, t, wt)
        if name == "combined":
            return lambda pred, target: (
                nn.CrossEntropyLoss()(pred, target) + self._generalized_dice_loss(pred, target)
            )
        raise ValueError(name)

    # ----------- differentiable losses -------------
    @staticmethod
    def _dice_loss(pred, target, smooth=1.0):
        pred = F.softmax(pred, 1)
        tgt = F.one_hot(target, pred.shape[1]).permute(0, 3, 1, 2).float()
        inter = (pred * tgt).sum((2, 3))
        union = pred.sum((2, 3)) + tgt.sum((2, 3))
        return 1 - ((2 * inter + smooth) / (union + smooth)).mean()

    @staticmethod
    def _generalized_dice_loss(pred, target, weight_type="square", smooth=1e-5):
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
        probs = F.softmax(pred, 1)
        tgt = F.one_hot(target, pred.shape[1]).permute(0, 3, 1, 2).float()
        cs = tgt.sum((0, 2, 3)).clamp(min=smooth)
        if weight_type == "square":
            w = 1 / (cs**2)
        elif weight_type == "simple":
            w = 1 / cs
        else:
            w = torch.ones_like(cs)
        inter = (probs * tgt).sum((0, 2, 3))
        union = (probs + tgt).sum((0, 2, 3))
        dice = (2 * (w * inter).sum() + smooth) / ((w * union).sum() + smooth)
        return 1 - dice

    # ---------------- optimizer & scheduler ----------------
    def configure_optimizers(self):
        opt_name = self.cfg.training.optimizer.name.lower()
        kw = dict(lr=self.cfg.training.optimizer.lr, weight_decay=self.cfg.training.optimizer.get("weight_decay", 0.0))
        if opt_name == "adam":
            opt = Adam(self.parameters(), **kw)
        elif opt_name == "adamw":
            opt = AdamW(self.parameters(), **kw)
        elif opt_name == "sgd":
            kw["momentum"] = self.cfg.training.optimizer.get("momentum", 0.9)
            opt = SGD(self.parameters(), **kw)
        else:
            raise ValueError(opt_name)

        if not self.cfg.training.get("scheduler", {}).get("enabled", False):
            return opt

        scheduler_params = self.cfg.training.scheduler
        if scheduler_params.name.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=scheduler_params.get("t_max", self.cfg.training.trainer.max_epochs),
                eta_min=scheduler_params.get("eta_min", 0),
            )
        elif scheduler_params.name.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                opt,
                step_size=scheduler_params.get("step_size", 10),
                gamma=scheduler_params.get("gamma", 0.1),
            )
        elif scheduler_params.name.lower() == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                patience=scheduler_params.get("patience", 5),
                factor=scheduler_params.get("factor", 0.1),
                mode=scheduler_params.get("mode", "min"),
            )
            return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": self.cfg.training.monitor_on}
        else:
            raise ValueError(f"Scheduler '{scheduler_params.name}' not recognized.")

        return {"optimizer": opt, "lr_scheduler": scheduler}

    # ---------------- step hooks ------------------
    def _common_step(self, batch, metrics, phase):
        imgs, gt = batch
        logits = self(imgs)
        loss = self.criterion(logits, gt)

        preds = logits.argmax(1).detach()
        metrics.update(preds, gt.detach())
        self.log(f"{phase}/loss", loss, on_step=True, prog_bar=(phase == "train"))
        if self.current_epoch % self.vis_every == 0 or self.current_epoch == self.trainer.max_epochs - 1:
            self._store_vis(phase, imgs, gt, preds)
        return loss

    def training_step(self, batch, _):
        return self._common_step(batch, self.train_metrics, "train")

    def validation_step(self, batch, _):
        self._common_step(batch, self.val_metrics, "val")

    def test_step(self, batch, _):
        return self._common_step(batch, self.test_metrics, "test")

    def predict_step(self, batch, _):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return self(x).argmax(1)

    def forward(self, x):
        return self.model(x)

    # -------------- epoch hooks --------------
    def on_train_epoch_end(self):
        self._emit(self.train_metrics)
        self._log_vis_grid("train")

    def on_validation_epoch_end(self):
        self._emit(self.val_metrics)
        self._log_vis_grid("val")

    def on_test_epoch_end(self):
        self._emit(self.test_metrics)
        self._log_vis_grid("test")

    # -------------- fit hooks --------------
    def on_fit_start(self):
        self._log_legend_once()

    # def on_fit_end(self):
    #     self._log_metrics_table("train")
    #     self._log_metrics_table("val")

    def on_test_end(self):
        # self._log_metrics_table("test")
        for metric in self.trainer.callback_metrics:
            if not metric.startswith(("test/", "test_class/")):
                continue
            self.test_results[metric] = round(float(self.trainer.callback_metrics[metric]), 4)

    # ---------------- metric logging --------------
    @staticmethod
    def safe_macro_mean(vec: torch.Tensor) -> torch.Tensor:
        if torch.isnan(vec).any():
            raise ValueError("Metric vector contains NaN - investigate upstream!")
        present = vec.ge(0)
        return vec[present].mean() if present.any() else torch.tensor(-1, device=vec.device)

    @torch.no_grad()
    def _emit(self, coll: MetricCollection):
        out = coll.compute()
        C = self.cfg.data.num_classes
        for full_key, tensor in out.items():
            phase, metric_name = full_key.split("/", 1)
            if tensor.ndim == 1 and tensor.numel() == C:
                mean_val = self.safe_macro_mean(tensor)
                self.log(f"{phase}/{metric_name}", mean_val, on_step=False, on_epoch=True)
                for i, val in enumerate(tensor):
                    self.log(
                        f"{phase}_class/{metric_name}_{get_class_name(i, self.cfg)}",
                        val if torch.isfinite(val) else -1.0,
                        on_step=False,
                        on_epoch=True,
                    )
            else:
                self.log(full_key, tensor, on_step=False, on_epoch=True)
        coll.reset()

    def _log_metrics_table(self, phase):
        # BUG: wandb logging issue
        logger.debug(f"Logging {phase} metrics table at epoch {self.current_epoch}")
        if not isinstance(self.logger, WandbLogger) or not self.trainer.is_global_zero:
            return

        C = self.cfg.data.num_classes

        cols = ["class"] + [
            x.split("/", 1)[1]
            for x in self.trainer.callback_metrics.keys()
            if x.startswith(f"{phase}/") and not x.endswith("_step")
        ]
        rows = []
        for c in range(C):
            name = get_class_name(c, self.cfg)
            row = [name]
            for n in cols[1:]:
                row.append(float(self.trainer.callback_metrics.get(f"{phase}_class/{n}_{name}", -1.0)))
            rows.append(row)
        mean_row = ["mean"]
        for n in cols[1:]:
            mean_row.append(float(self.trainer.callback_metrics.get(f"{phase}/{n}", -1.0)))
        rows.append(mean_row)
        table = wandb.Table(columns=cols, data=rows)
        self.logger.experiment.log({f"{phase}_metrics_table": table}, step=self.current_epoch)
        logger.debug(f"{phase} metrics table logged at epoch {self.current_epoch}")

    # ---------------- visualisation ----------------
    def _log_legend_once(self):
        if self._legend_logged or not self.trainer.is_global_zero:
            return
        legend_img = self._build_color_legend()
        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(
                key="samples/legend", images=[wandb.Image(legend_img)], caption=["Color legend"], step=0
            )
        elif isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image("samples/legend", legend_img, global_step=0)
        self._legend_logged = True

    def _build_color_legend(self):
        C, _ = self.color_lut.shape
        block = 32
        pad = 6

        bar = torch.zeros(3, C * block + pad, block)
        for i, colour in enumerate(self.color_lut):
            bar[:, pad + i * block : pad + (i + 1) * block, :] = colour.view(3, 1, 1)

        fig, ax = plt.subplots(figsize=(2.4, 0.25 * C + 0.3))
        ax.imshow(bar.permute(1, 2, 0))
        ax.axis("off")

        for i in range(C):
            name = get_class_name(i, self.cfg)
            y = pad + (i + 0.5) * block
            ax.text(
                block + 4,
                y,
                name,
                va="center",
                ha="left",
                fontsize=12,
                color="black",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
            )

        fig.tight_layout(pad=0)
        return fig

    def _store_vis(self, phase, imgs, gt, preds):
        self.vis_samples[phase] = (
            imgs[: self.max_vis].cpu(),
            gt[: self.max_vis].cpu(),
            preds[: self.max_vis].cpu(),
        )

    def _colorize(self, mask: torch.Tensor):
        h, w = mask.shape
        rgb = self.color_lut[mask.view(-1)].view(h, w, 3).permute(2, 0, 1)
        return rgb

    @torch.no_grad()
    def _log_vis_grid(self, phase):
        """Create a 4-panel grid per sample and log via W&B / TensorBoard."""
        if phase not in self.vis_samples:
            return
        imgs, gt, preds = self.vis_samples.pop(phase)
        panels = []
        for img, gt, pr in zip(imgs, gt, preds):
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            diff = torch.zeros_like(img)
            diff[0][gt != pr] = 1.0  # red wrong
            diff[1][gt == pr] = 1.0  # green correct
            overlay = img * (1 - self.overlay_alpha) + diff * self.overlay_alpha
            panels.extend([img.clamp(0, 1), self._colorize(gt), self._colorize(pr), overlay.clamp(0, 1)])
        grid = make_grid(panels, nrow=4)
        caption = f"epoch_{self.current_epoch}: Image | Ground Truth | Prediction | Overlay"
        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(key=f"samples/{phase}", images=[grid], caption=[caption], step=self.current_epoch)
        elif isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image(f"{phase}_seg_samples", grid, global_step=self.current_epoch)
