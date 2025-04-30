"""
Abstract base class for all Mars surface image classification models.
"""

import io
import logging
import textwrap
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.sgd import SGD
from torchmetrics import AUROC
from torchmetrics import Accuracy
from torchmetrics import F1Score
from torchmetrics import MetricCollection
from torchmetrics import Precision
from torchmetrics import Recall
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid

logger = logging.getLogger(__name__)


class BaseClassificationModel(LightningModule, ABC):
    """Abstract base class for classification models."""

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

        # MetricCollection (single base cloned per phase)
        subtask = cfg.data.subtask
        num_classes = cfg.data.num_classes
        metric_args = {"task": subtask}
        if subtask == "multilabel":
            metric_args["num_labels"] = num_classes
        else:
            metric_args["num_classes"] = num_classes

        # Create metrics with appropriate arguments
        metrics_dict = {}
        for average in ["macro", "weighted", "none"]:
            suffix = "_" + average if average != "none" else "_per_class"
            metrics_dict["Accuracy" + suffix] = Accuracy(**metric_args, average=average)
            metrics_dict["Precision" + suffix] = Precision(**metric_args, average=average)
            metrics_dict["Recall" + suffix] = Recall(**metric_args, average=average)
            metrics_dict["F1Score" + suffix] = F1Score(**metric_args, average=average)
            metrics_dict["AUROC" + suffix] = AUROC(**metric_args, average=average)

        base = MetricCollection(metrics_dict)

        self.train_metrics = base.clone(prefix="train/")
        self.val_metrics = base.clone(prefix="val/")
        self.test_metrics = base.clone(prefix="test/")

        # visual logging
        self.vis_every = cfg.logger.get("vis_every", 3)
        self.max_vis = cfg.logger.get("max_vis_samples", 4)
        # samples stored as {phase: (imgs, ground_truth, preds)}
        self.vis_samples: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        cmap = plt.get_cmap("tab20", num_classes)
        self.color_lut = torch.tensor([cmap(i)[:3] for i in range(num_classes)])  # [C,3]

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
    def _initialize_model(self):
        """Initialize the model architecture. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _initialize_model method.")

    # ---------------- criterion ---------------------
    def _initialize_criterion(self):
        criterion_name = self.cfg.training.criterion.name
        if criterion_name in ["cross_entropy", "bce"]:
            if self.cfg.data.subtask == "multiclass":
                if criterion_name == "bce":
                    logger.warning("Using CrossEntropyLoss instead of BCEWithLogitsLoss for multiclass classification.")
                return nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            else:
                return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Criterion '{criterion_name}' not recognized.")

    # ---------------- optimizer ---------------------
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

    # ---------------- step hooks ------------------
    def _common_step(self, batch, batch_idx, metrics, phase):
        imgs, ground_truth = batch
        logits = self(imgs)
        loss = self.criterion(logits, ground_truth)

        metrics.update(logits.detach(), ground_truth.detach())
        self.log(f"{phase}/loss", loss, on_step=True, prog_bar=True)
        if (
            self.current_epoch % self.vis_every == 0 or self.current_epoch == self.trainer.max_epochs - 1
        ) and batch_idx == 0:
            if self.cfg.data.subtask == "multiclass":
                probs = F.softmax(logits, dim=1)
                preds = probs.argmax(1)
            else:
                probs = F.sigmoid(logits)
                preds = (probs > 0.5).long()
            self._store_vis(phase, imgs, ground_truth, probs, preds)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, self.train_metrics, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, self.val_metrics, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, self.test_metrics, "test")

    def predict_step(self, batch, _):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        logits = self(x)
        if self.cfg.data.subtask == "multiclass":
            preds = logits.argmax(1)
        else:
            preds = (F.sigmoid(logits) > 0.5).long()
        return preds

    def forward(self, x):
        return self.model(x)

    # ---------------- epoch hooks --------------
    def on_train_epoch_end(self):
        self._emit(self.train_metrics)
        self._log_vis_grid("train")

    def on_validation_epoch_end(self):
        self._emit(self.val_metrics)
        self._log_vis_grid("val")

    def on_test_epoch_end(self):
        self._emit(self.test_metrics)
        self._log_vis_grid("test")

    # ---------------- fit hooks --------------
    def on_test_end(self):
        for metric in self.trainer.callback_metrics:
            if not metric.startswith(("test/", "test_class/")):
                continue
            self.test_results[metric] = round(float(self.trainer.callback_metrics[metric]), 4)

    # ---------------- metric logging --------------
    def _get_class_name(self, class_idx):
        """Get human-readable class name for given class index."""
        return class_idx if not self.cfg.get("mapping") else self.cfg.mapping.get(class_idx, class_idx)

    def _emit(self, coll: MetricCollection):
        out = coll.compute()
        for full_key, tensor in out.items():
            phase, metric_name = full_key.split("/", 1)
            if metric_name.endswith("_class"):
                metric_name = metric_name.replace("_per_class", "")
                for i, val in enumerate(tensor):
                    self.log(
                        f"{phase}_class/{metric_name}_{self._get_class_name(i)}",
                        val if torch.isfinite(val) else -1.0,
                        on_step=False,
                        on_epoch=True,
                    )
            else:
                self.log(full_key, tensor, on_step=False, on_epoch=True)
        coll.reset()

    # ---------------- visualisation ----------------
    def _store_vis(self, phase, imgs, ground_truth, probs, preds):
        self.vis_samples[phase] = (
            imgs[: self.max_vis].cpu(),
            ground_truth[: self.max_vis].cpu(),
            probs[: self.max_vis].cpu(),
            preds[: self.max_vis].cpu(),
        )

    @staticmethod
    def bordered(im: torch.Tensor, correct: bool, h_border: int) -> torch.Tensor:
        colour = torch.tensor([0, 1, 0] if correct else [1, 0, 0], dtype=im.dtype, device=im.device)[:, None, None]
        framed = im.clone()
        framed[:, :h_border] = colour
        framed[:, -h_border:] = colour
        framed[:, :, :h_border] = colour
        framed[:, :, -h_border:] = colour
        out = framed.clamp(0, 1)
        return out

    @staticmethod
    def text_panel(lines: str, width: int, height: int, wrap: int, font_size: int) -> torch.Tensor:
        """Render wrapped text on white background."""
        wrapped = textwrap.fill(lines, wrap)
        txt_img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(txt_img)
        draw.text((3, 3), wrapped, fill=(0, 0, 0), font=ImageFont.truetype("DejaVuSans.ttf", font_size))
        out = to_tensor(txt_img)
        return out

    def prob_panel(self, probs: torch.Tensor, topk: int, width: int, height: int) -> torch.Tensor:
        """Return a torch tensor with a tiny matplotlib plot."""
        # Pick plotting style per task
        if self.cfg.data.subtask == "binary":
            classes = (
                ["neg", "pos"]
                if not self.cfg.data.get("mapping")
                else [self.cfg.data.mapping.get(0, "neg"), self.cfg.data.mapping.get(1, "pos")]
            )
            values = probs.cpu().numpy().tolist()
        elif self.cfg.data.subtask == "multiclass":
            values, idxs = probs.topk(min(topk, probs.numel()))
            classes = [self._get_class_name(i.item()) for i in idxs]
            values = values.cpu().numpy().tolist()
        else:  # multilabel
            classes = [self._get_class_name(i) for i in range(probs.numel())]
            values = probs.cpu().numpy().tolist()
        # ---- plot ----
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        if self.cfg.data.subtask == "multilabel":
            ax.barh(classes, values)
        else:
            ax.bar(classes, values)
        ax.set(ylim=(-0.1 if self.cfg.data.subtask == "multilabel" else None, 1.05), yticks=[], xlabel="")
        ax.tick_params(axis="x", labelsize=18)
        ax.margins(x=0)
        fig.tight_layout(pad=0.1)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=False)
        plt.close(fig)
        buf.seek(0)
        out = to_tensor(Image.open(buf).convert("RGB"))
        return out

    @torch.no_grad()
    def _log_vis_grid(self, phase, topk: int = 5, h_border: int = 2, font_size: int = 24, wrap: int = 32):
        """
        Build a 4x3 grid per epoch:
            ┌─────────┬────────────┬───────────────┐
            │ image   │ GT / Pred  │ prob visual   │
            ├─────────┼────────────┼───────────────┤
            │   …     │     …      │      …        │
            └─────────┴────────────┴───────────────┘
        and push to WandB / TensorBoard.
        """
        if phase not in self.vis_samples:
            return

        imgs, ground_truth, probs, preds = self.vis_samples.pop(phase)
        task = self.cfg.data.subtask

        panels = []
        base_w, base_h = self.cfg.model.input_size[-2:]

        for img, gt, pr, prob_vec in zip(imgs, ground_truth, preds, probs):
            if img.shape[0] == 1:  # grayscale → RGB
                img = img.repeat(3, 1, 1)

            correct = (gt == pr).all() if task == "multilabel" else gt == pr
            img_cell = self.bordered(img, correct, h_border)

            # --- column 2 text ------------------------------------------------
            if task == "binary":
                txt = f"GT {gt.item()}  |  Pred {pr.item()}  p={prob_vec:.2f}"
            elif task == "multiclass":
                txt = f"GT {self._get_class_name(gt.item())} | " f"Pred {self._get_class_name(pr.item())}"
            else:
                act_gt = torch.where(gt == 1)[0].tolist()
                act_pr = torch.where(pr == 1)[0].tolist()
                txt = f"GT {act_gt} | Pred {act_pr}"
            txt_cell = self.text_panel(txt, base_w, base_h, wrap, font_size)

            # --- column 3 prob vis -------------------------------------------
            prob_cell = self.prob_panel(prob_vec, topk, base_w, base_h)
            panels.extend([img_cell, txt_cell, prob_cell])

        grid = make_grid(panels, nrow=3)  # 3 columns

        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(
                key=f"samples/{phase}",
                images=[grid],
                caption=[""],
                step=self.current_epoch,
            )
        elif isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image(f"{phase}_cls_samples", grid, global_step=self.current_epoch)
