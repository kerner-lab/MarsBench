"""Utility functions for testing classification and segmentation models."""

import tempfile
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
VALID_TASKS = {"classification", "segmentation"}


class ModelTestDataset(Dataset):
    """Dataset for generating dummy data for model testing."""

    def __init__(self, input_size: Tuple[int, ...], num_classes: int, task: str):
        if task not in VALID_TASKS:
            raise ValueError(f"Task must be one of {VALID_TASKS}, got {task}")
        self.input_size = input_size
        self.num_classes = num_classes
        self.task = task

    def __len__(self) -> int:
        return DEFAULT_DATASET_SIZE

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a random input tensor and target for testing."""
        dummy_input = torch.randn(*self.input_size)
        if self.task == "classification":
            dummy_label = torch.randint(0, self.num_classes, (1,)).item()
        else:  # segmentation
            H, W = self.input_size[1:]
            dummy_label = torch.randint(0, self.num_classes, (H, W))
        return dummy_input, dummy_label


def create_test_data(
    batch_size: int = DEFAULT_BATCH_SIZE,
    input_size: Tuple[int, ...] = DEFAULT_INPUT_SIZE,
    num_classes: int = 2,
    task: str = "classification",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Creates input and target tensors for testing models."""
    """Create test input and target tensors."""
    dummy_input = torch.randn(batch_size, *input_size, requires_grad=True)

    if task == "classification":
        dummy_target = torch.randint(0, num_classes, (batch_size,))
    else:  # segmentation
        H, W = input_size[1:]
        dummy_target = torch.randint(0, num_classes, (batch_size, H, W))

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
        probs = F.softmax(output, dim=1)
        assert torch.all(
            (probs >= 0) & (probs <= 1)
        ), f"Model {model_name} produced invalid probabilities"
        assert torch.allclose(
            probs.sum(dim=1), torch.ones_like(probs.sum(dim=1))
        ), f"Model {model_name} probabilities don't sum to 1"


def setup_training(
    model: torch.nn.Module,
    input_size: Tuple[int, ...],
    num_classes: int,
    task: str,
    batch_size: int,
    max_epochs: int,
) -> None:
    """Setup and run training loop."""
    dataset = ModelTestDataset(input_size, num_classes, task)
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    val_dataloader = DataLoader(dataset, batch_size=batch_size)

    trainer = Trainer(max_epochs=max_epochs, fast_dev_run=True)
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


def verify_model_save_load(
    model: torch.nn.Module, model_class: Any, cfg: Any, model_name: str
) -> None:
    """Verify model can be saved and loaded correctly."""
    with tempfile.NamedTemporaryFile() as tmp:
        torch.save(model.state_dict(), tmp.name)
        model_loaded = model_class(cfg)
        model_loaded.load_state_dict(torch.load(tmp.name, weights_only=False))

    for param_original, param_loaded in zip(
        model.parameters(), model_loaded.parameters()
    ):
        assert torch.allclose(
            param_original, param_loaded
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
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Criterion '{criterion_name}' not recognized.")

    loss = criterion(output, target)
    loss.backward()

    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    assert (
        grad_norm > 0
    ), f"{model_name}: Gradients are not computed during backward pass."
