"""Utility functions for testing classification andand segmn"""

import tempfile
from token import OP
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# Constants for testing
DEFAULT_BATCH_SIZE = 2
DEFAULT_NUM_EPOCHS = 1
DEFAULT_DATASET_SIZE = 10
DEFAULT_INPUT_SIZE = (3, 224, 224)
VALID_TASKS = {"classification", "segmentation", "detection"}


class ModelTestDataset(Dataset):
    """Dataset for generating dummy data for model testing."""

    def __init__(
        self,
        input_size: Tuple[int, ...],
        num_classes: int,
        task: str,
        class_idx_offset: Optional[int] = None,
    ):
        if task not in VALID_TASKS:
            raise ValueError(f"Task must be one of {VALID_TASKS}, got {task}")
        self.input_size = input_size
        self.num_classes = num_classes
        self.task = task
        self.class_idx_offset = class_idx_offset

    def __len__(self) -> int:
        return DEFAULT_DATASET_SIZE

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a random input tensor and target for testing."""
        # Create input with expected shape for the model
        dummy_input = torch.randn(*self.input_size)

        if self.task == "classification":
            # For classification, target is a class index
            dummy_label = torch.randint(0, self.num_classes, (1,)).item()
        elif self.task == "segmentation":
            # For segmentation, target is a 2D mask with class indices
            H, W = self.input_size[1:]
            dummy_label = torch.randint(0, self.num_classes, (H, W))
        else:  # detection
            H, W = self.input_size[1:]
            boxes = torch.tensor([[10.0, 15.0, W - 10.0, H - 10.0]])
            labels = torch.tensor([self.class_idx_offset])
            dummy_label = {
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor([idx]),
                "img_size": torch.tensor([H, W]),
                "img_scale": torch.tensor([1.0]),
            }
        return dummy_input, dummy_label


def create_test_data(
    batch_size: int = DEFAULT_BATCH_SIZE,
    input_size: Tuple[int, ...] = DEFAULT_INPUT_SIZE,
    num_classes: int = 2,
    task: str = "classification",
    model_name: Optional[str] = None,
    bbox_format: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Creates input and target tensors for testing models."""
    dummy_input = torch.randn(batch_size, *input_size, requires_grad=True)

    if task == "classification":
        dummy_target = torch.randint(0, num_classes, (batch_size,))
    elif task == "segmentation":
        H, W = input_size[1:]
        dummy_target = torch.randint(0, num_classes, (batch_size, H, W))
    else:  # detection
        dummy_target = []
        for i in range(batch_size):
            H, W = input_size[1:]
            num_boxes = torch.randint(1, 2, (1,)).item()
            xmin = torch.randint(0, W - 20, (num_boxes,)).float()
            ymin = torch.randint(0, H - 20, (num_boxes,)).float()
            xmax = xmin + torch.randint(10, min(W - xmin.int().min().item(), W - 10), (num_boxes,)).float()
            ymax = ymin + torch.randint(10, min(H - ymin.int().min().item(), H - 10), (num_boxes,)).float()
            boxes = torch.stack([xmin, ymin, xmax, ymax], dim=1)

            if bbox_format == "coco":
                boxes[:, 2] = xmax - xmin
                boxes[:, 3] = ymax - ymin
                labels = torch.ones((num_boxes,), dtype=torch.long)
            elif bbox_format == "yolo":
                cx = (xmin + xmax) / 2 / W
                cy = (ymin + ymax) / 2 / H
                w = (xmax - xmin) / W
                h = (ymax - ymin) / H
                boxes = torch.stack([cx, cy, w, h], dim=1)
                labels = torch.zeros((num_boxes,), dtype=torch.long)
            else:
                labels = torch.ones((num_boxes,), dtype=torch.long)

            if model_name.lower() == "efficientdet":
                boxes = boxes[:, [1, 0, 3, 2]]  # yxyx

            dummy_target.append(
                {
                    "boxes": boxes,
                    "labels": labels,
                    "image_id": torch.tensor([i]),
                    "img_size": torch.tensor([H, W]),
                    "img_scale": torch.tensor([1.0]),
                }
            )
    return dummy_input, dummy_target


def get_expected_output_shape(
    batch_size: int, num_classes: int, input_size: Tuple[int, ...], task: str
) -> Tuple[int, ...]:
    """Get expected output shape based on task."""
    if task == "classification":
        return (batch_size, num_classes)
    else:  # segmentation
        H, W = input_size[1:]
        return (batch_size, num_classes, H, W)


def verify_output_properties(output: torch.Tensor, task: str, model_name: str) -> None:
    """Checks if model outputs are valid (no inf/nan, proper probabilities)."""
    """Verify model output properties."""
    # Check for invalid values
    assert not torch.isinf(output).any(), f"Model {model_name} produced infinite values"
    assert not torch.isnan(output).any(), f"Model {model_name} produced NaN values"

    # For segmentation, verify probability distribution
    if task == "segmentation":
        # Reshape output to (batch_size, num_classes, H, W) if necessary
        if len(output.shape) == 2:
            output = output.view(-1, output.shape[1], 1, 1)
        probs = F.softmax(output, dim=1)
        assert torch.all((probs >= 0) & (probs <= 1)), f"Model {model_name} produced invalid probabilities"
        assert torch.allclose(
            probs.sum(dim=1), torch.ones_like(probs.sum(dim=1))
        ), f"Model {model_name} probabilities don't sum to 1"


def detection_collate_fn(batch):
    images, targets = tuple(zip(*batch))
    images = torch.stack(images, dim=0)
    return images, targets


def detection_collate_fn_v2(batch):
    images, targets = tuple(zip(*batch))
    images = torch.stack(images, dim=0)

    boxes = [target["boxes"][:, [1, 0, 3, 2]] for target in targets]
    labels = [target["labels"] for target in targets]
    img_sizes = torch.stack([target["img_size"] for target in targets])
    img_scales = torch.tensor([target["img_scale"] for target in targets])

    annotations = {
        "bbox": boxes,
        "cls": labels,
        "img_size": img_sizes,
        "img_scale": img_scales,
    }
    return images, annotations


def setup_training(
    model: torch.nn.Module,
    input_size: Tuple[int, ...],
    num_classes: int,
    task: str,
    batch_size: int,
    max_epochs: int,
    model_name: str = None,
    bbox_format: str = None,
) -> None:
    """Setup and run training loop."""

    class_idx_offset = None
    if task == "detection":
        if bbox_format == "yolo":
            class_idx_offset = 0
        else:
            class_idx_offset = 1
        if model_name.lower() == "efficientdet":
            collate_fn = detection_collate_fn_v2
        else:
            collate_fn = detection_collate_fn
    else:
        collate_fn = None

    dataset = ModelTestDataset(input_size, num_classes, task, class_idx_offset)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    trainer = Trainer(max_epochs=max_epochs, fast_dev_run=True)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


def verify_model_save_load(model: torch.nn.Module, model_class: Any, cfg: Any, model_name: str) -> None:
    """Verify model can be saved and loaded correctly."""
    with tempfile.NamedTemporaryFile() as tmp:
        torch.save(model.state_dict(), tmp.name)
        model_loaded = model_class(cfg)
        model_loaded.load_state_dict(torch.load(tmp.name, weights_only=False))

    for param_original, param_loaded in zip(model.parameters(), model_loaded.parameters()):
        assert torch.allclose(
            param_original.cpu(), param_loaded.cpu()
        ), f"{model_name}: Parameters differ after loading."


def verify_backward_pass(
    model: torch.nn.Module,
    output: torch.Tensor,
    target: torch.Tensor,
    criterion_name: str,
    model_name: str,
) -> None:
    """Verify model backward pass."""
    if criterion_name == "cross_entropy":
        # For segmentation, target shape needs to be modified for CrossEntropyLoss
        if len(output.shape) == 4:  # This is a segmentation model output (B, C, H, W)
            # CrossEntropyLoss for segmentation expects target as (B, H, W) with class indices
            if len(target.shape) == 3:  # Target is already (B, H, W)
                criterion = torch.nn.CrossEntropyLoss()
            else:
                # If target is not the right shape, we need to reshape it
                raise ValueError(f"Unexpected target shape for segmentation: {target.shape}")
        else:  # Classification
            criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Criterion '{criterion_name}' not recognized.")

    # Before computing loss, ensure output and target have compatible dimensions
    if len(output.shape) == 4 and output.shape[1] != target.shape[0]:
        # This is likely the segmentation tensor size mismatch issue
        # Reshape target to have the same spatial dimensions as output if needed
        if len(target.shape) == 3:  # (B, H, W)
            # No reshaping needed for (B, H, W) format
            pass
        elif len(target.shape) == 4:  # (B, C, H, W)
            # If target is already 4D, ensure it has the right number of classes
            if target.shape[1] != output.shape[1]:
                # Critical mismatch - the number of classes in target doesn't match output
                raise ValueError(
                    f"Output has {output.shape[1]} classes but target has {target.shape[1]} dimensions. "
                    "For segmentation, ensure target format matches expected input for CrossEntropyLoss."
                )

    loss = criterion(output, target)
    loss.backward()

    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, f"{model_name}: Gradients are not computed during backward pass."
