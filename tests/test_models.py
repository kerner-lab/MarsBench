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
from tests.utils.model_test_utils import verify_backward_pass_detection
from tests.utils.model_test_utils import verify_backward_pass_detection_detr
from tests.utils.model_test_utils import verify_backward_pass_detection_efficientdet
from tests.utils.model_test_utils import verify_detection_output_properties
from tests.utils.model_test_utils import verify_detr_output_properties
from tests.utils.model_test_utils import verify_efficientdet_output_properties
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


def verify_detection_model_output_and_backward_pass(
    model, model_name, dummy_input, dummy_target, cfg
):
    device = next(model.parameters()).device
    training_mode = model.training
    dummy_target = [{k: v.to(device) for k, v in t.items()} for t in dummy_target]
    print("check3")

    if model_name.lower() == "detr":
        print("check4")
        dummy_input = dummy_input.detach()
        dummy_input = dummy_input.to(device)

        output = model(dummy_input)
        print("check5")
        verify_detr_output_properties(output, model_name)
        print("check6")
        verify_backward_pass_detection_detr(model, output, dummy_target, model_name)
        print("check7")

    elif model_name.lower() == "efficientdet":
        dummy_input = dummy_input.to(device)

        dummy_target = {
            "bbox": [target["boxes"] for target in dummy_target],
            "cls": [target["labels"] for target in dummy_target],
            "img_size": torch.stack([target["img_size"] for target in dummy_target]),
            "img_scale": torch.tensor([target["img_scale"] for target in dummy_target]),
        }

        model.eval()
        with torch.no_grad():
            output = model(dummy_input, dummy_target)
        model.train(training_mode)

        expected_eval_keys = {"loss", "class_loss", "box_loss", "detections"}
        assert expected_eval_keys.issubset(
            output.keys()
        ), f"{model_name}: Evaluation output missing keys. Expected {expected_eval_keys}, got {output.keys()}"

        detections = output["detections"]
        verify_efficientdet_output_properties(detections, model_name)

        loss_dict = model(dummy_input, dummy_target)
        verify_backward_pass_detection_efficientdet(
            model, loss_dict["loss"], model_name
        )

    else:
        print("checkerror")
        dummy_input = dummy_input.to(device)

        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        model.train(training_mode)

        verify_detection_output_properties(output, model_name)

        loss_dict = model(dummy_input, dummy_target)
        verify_backward_pass_detection(model, loss_dict, model_name)


def run_detection_model_tests(
    cfg, model, model_class, model_name, input_size, batch_size
):
    dummy_input, dummy_target = create_test_data(
        batch_size=batch_size,
        input_size=input_size,
        num_classes=cfg.data.num_classes,
        task="detection",
        model_name=model_name,
        bbox_format=cfg.model.bbox_format,
    )
    print("check2")

    verify_detection_model_output_and_backward_pass(
        model, model_name, dummy_input, dummy_target, cfg
    )
    print("check2a")

    setup_training(
        model=model,
        input_size=input_size,
        num_classes=cfg.data.num_classes,
        task="detection",
        batch_size=cfg.training.batch_size,
        max_epochs=cfg.training.trainer.max_epochs,
        model_name=model_name,
        bbox_format=cfg.model.bbox_format,
    )
    print(f"{model_name}: Training integration test successful")

    verify_model_save_load(model, model_class, cfg, model_name)
    print(f"{model_name}: Model saving and loading successful")


@pytest.mark.parametrize(
    "model_config_file",
    glob.glob("configs/model/**/*.yaml", recursive=True),
    ids=lambda x: os.path.splitext(os.path.basename(x))[0],
)
def test_models(model_config_file: str) -> None:
    """Tests model initialization, forward/backward passes, and training loop."""
    # Extract task and model name from the config file path
    rel_path = os.path.relpath(model_config_file, "configs/model")
    task = rel_path.split(os.sep)[0]  # classification or segmentation
    model = os.path.splitext(os.path.basename(model_config_file))[0]

    config_dir = os.path.abspath("configs")

    # Initialize Hydra and compose configuration
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        # Set default dataset based on task
        default_dataset = "domars16k" if task == "classification" else "cone_quest"

        # Set appropriate number of classes based on task
        if task == "classification":
            num_classes = 10  # Classification models use 10 classes for testing
        elif task == "segmentation":
            # Set to 8 classes for all segmentation models to ensure consistency
            num_classes = 8  # Use 8 classes for all segmentation models
        else:
            num_classes = 2

        cfg = compose(
            config_name="config",
            overrides=[
                f"task={task}",
                f"model={task}/{model}",
                f"data={task}/{default_dataset}",
                f"data.num_classes={num_classes}",  # Use consistent class count
            ],
        )

    # Check model status
    if (
        not hasattr(cfg.model, "status")
        or cfg.model.status not in cfg.test.model.status
    ):
        print(
            f"Skipping model '{model}' for {task} (status: {getattr(cfg.model, 'status', 'unknown')})"
        )
        pytest.skip(f"Model '{model}' for {task} is not ready for testing.")

    print(f"Testing model '{model}' for {task}")
    model_class_path = cfg.model.get("class_path", None)
    if model_class_path is None:
        pytest.fail(
            f"Model class path not specified for model '{model}' in the configuration."
        )

    # Import model class
    ModelClass = import_model_class(model_class_path, model)
    model_name = cfg.model.name

    # Setup model parameters
    input_size = cfg.model.get("input_size", [3, 224, 224])
    batch_size = 2
    model = ModelClass(cfg)
    model.train()

    if task == "detection":
        print("check1")
        print(model_name)
        run_detection_model_tests(
            cfg=cfg,
            model=model,
            model_class=ModelClass,
            model_name=model_name,
            input_size=input_size,
            batch_size=batch_size,
        )
        # run_detection_model_tests(
        #         cfg, model, ModelClass, model_name, input_size, batch_size
        # )

    else:
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
        if (
            isinstance(output, tuple)
            and cfg.model.name in cfg.test.model.with_tuple_output
        ):
            output = output[0]
        elif isinstance(output, tuple):
            pytest.fail(f"Not expecting tuple as output for Model: '{model}'.")

        assert (
            output.shape == expected_output_shape
        ), f"{model}: Expected output shape {expected_output_shape}, got {output.shape}"

        # Verify output properties
        verify_output_properties(output, task, model)
        print(f"{model}: Forward pass successful with output shape {output.shape}")

        # Test backward pass
        verify_backward_pass(
            model, output, dummy_target, cfg.training.criterion.name, model
        )
        print(f"{model}: Backward pass successful")

        # Test training loop
        setup_training(
            model=model,
            input_size=input_size,
            num_classes=cfg.data.num_classes,
            task=task,
            batch_size=cfg.training.batch_size,
            max_epochs=cfg.training.trainer.max_epochs,
        )
        print(f"{model}: Training integration test successful")

        # Test model save/load
        verify_model_save_load(model, ModelClass, cfg, model)
        print(f"{model}: Model saving and loading successful")
