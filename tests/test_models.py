"""Tests for classification and segmentation model implementations."""

import glob
import importlib
import os
from typing import Any
from typing import Type

import pytest
import torch
from hydra import compose
from hydra import initialize_config_dir
from omegaconf import DictConfig

from src.models import *
from tests.utils.model_test_utils import DEFAULT_BATCH_SIZE
from tests.utils.model_test_utils import create_test_data
from tests.utils.model_test_utils import get_expected_output_shape
from tests.utils.model_test_utils import setup_training
from tests.utils.model_test_utils import verify_backward_pass
from tests.utils.model_test_utils import verify_model_save_load
from tests.utils.model_test_utils import verify_output_properties


def import_model_class(model_class_path: str, model_name: str) -> Type[torch.nn.Module]:
    """Imports and returns the model class from its path."""
    _, class_name = model_class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(model_class_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        pytest.fail(
            f"Failed to import model class '{model_class_path}' for model '{model_name}': {e}"
        )


def verify_model_config(cfg: DictConfig, task: str, model_name: str) -> None:
    """Checks if model configuration is valid for testing."""
    if cfg.model.get(task).status not in cfg.test.model.status:
        pytest.skip(f"Model '{model_name}' for {task} is not ready for testing.")

    if cfg.model.get(task, {}).get("class_path", None) is None:
        pytest.fail(
            f"Model class path not specified for model '{model_name}' in the configuration."
        )


@pytest.mark.parametrize(
    "model_config_file",
    glob.glob("configs/model/*.yaml"),
    ids=lambda x: os.path.splitext(os.path.basename(x))[0],
)
def test_models(model_config_file: str) -> None:
    """Tests model initialization, forward/backward passes, and training loop."""
    model = os.path.splitext(os.path.basename(model_config_file))[0]
    config_dir = os.path.abspath("configs")

    # Initialize Hydra and compose configuration
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"model={model}",
                "data.num_classes=10",
            ],
        )

    for task in cfg.model:
        model_name = cfg.model.get(task).name
        # Check model status
        if cfg.model.get(task).status not in cfg.test.model.status:
            print(
                f"Skipping model '{model_name}' for {task} (status: {cfg.model.get(task).status})"
            )
            pytest.skip(f"Model '{model_name}' for {task} is not ready for testing.")

        print(f"Testing model '{model_name}' for {task}")
        model_class_path = cfg.model.get(task, {}).get("class_path", None)
        if model_class_path is None:
            pytest.fail(
                f"Model class path not specified for model '{model_name}' in the configuration."
            )

        # Import model class
        ModelClass = import_model_class(model_class_path, model_name)

        # Setup model parameters
        input_size = cfg.model.get(task).get("input_size", [3, 224, 224])
        batch_size = 2
        model = ModelClass(cfg)
        model.train()

        # Create test data
        dummy_input, dummy_target = create_test_data(
            batch_size=batch_size,
            input_size=input_size,
            num_classes=cfg.data.num_classes,
            task=task,
        )

        # Test forward pass
        output = model(dummy_input)
        expected_output_shape = get_expected_output_shape(
            batch_size=batch_size,
            num_classes=cfg.data.num_classes,
            input_size=input_size,
            task=task,
        )

        # Handle tuple outputs
        if isinstance(output, tuple) and model_name in cfg.test.model.with_tuple_output:
            output = output[0]
        elif isinstance(output, tuple):
            pytest.fail(f"Not expecting tuple as output for Model: '{model_name}'.")

        assert (
            output.shape == expected_output_shape
        ), f"{model_name}: Expected output shape {expected_output_shape}, got {output.shape}"

        # Verify output properties
        verify_output_properties(output, task, model_name)
        print(f"{model_name}: Forward pass successful with output shape {output.shape}")

        # Test backward pass
        verify_backward_pass(
            model, output, dummy_target, cfg.criterion.name, model_name
        )
        print(f"{model_name}: Backward pass successful")

        # Test training loop
        setup_training(
            model=model,
            input_size=input_size,
            num_classes=cfg.data.num_classes,
            task=task,
            batch_size=cfg.training.batch_size,
            max_epochs=cfg.training.max_epochs,
        )
        print(f"{model_name}: Training integration test successful")

        # Test model save/load
        verify_model_save_load(model, ModelClass, cfg, model_name)
        print(f"{model_name}: Model saving and loading successful")
