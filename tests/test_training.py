"""Tests for the training module functions"""
import os
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from marsbench.training.callbacks import setup_callbacks
from marsbench.training.execution import run_prediction
from marsbench.training.execution import run_testing
from marsbench.training.execution import run_training
from marsbench.training.model_setup import setup_model
from marsbench.training.results import save_benchmark_results
from marsbench.training.results import save_predictions


def test_setup_callbacks():
    """Test that callbacks are set up correctly."""
    # Create a config structure that matches the expected format
    cfg = OmegaConf.create(
        {
            "training": {"early_stopping_patience": 3, "trainer": {"enable_checkpointing": True}},
            "callbacks": {
                "early_stopping": {"enabled": True, "monitor": "val/loss", "mode": "min"},
                "best_checkpoint": {"enabled": True, "monitor": "val/loss", "mode": "min", "save_top_k": 1},
                "last_checkpoint": {"enabled": True, "save_last": True},
            },
        }
    )

    # Run with minimal callbacks
    callbacks = setup_callbacks(cfg)

    # Verify we have callbacks
    assert len(callbacks) > 0

    # Verify with disabled callbacks
    for callback in cfg.callbacks:
        cfg.callbacks[callback].enabled = False

    callbacks = setup_callbacks(cfg)
    assert len(callbacks) == 0


def test_run_training():
    """Test the training execution function."""
    # Create test objects
    trainer = MagicMock()
    model = MagicMock()
    data_module = MagicMock()

    # Test without test_after_training
    cfg = OmegaConf.create({"test_after_training": False})
    run_training(trainer, model, data_module, cfg)

    # Verify trainer.fit was called
    trainer.fit.assert_called_once_with(model, data_module)

    # Test with test_after_training
    cfg = OmegaConf.create({"test_after_training": True})
    trainer.reset_mock()

    # Mock run_testing to avoid dependencies
    with patch("marsbench.training.execution.run_testing") as mock_run_testing:
        run_training(trainer, model, data_module, cfg)

        # Verify trainer.fit was called
        trainer.fit.assert_called_once_with(model, data_module)

        # Verify run_testing was called
        mock_run_testing.assert_called_once_with(trainer, model, data_module, cfg)


def test_run_testing():
    """Test the testing execution function."""
    # Create test objects
    trainer = MagicMock()
    model = MagicMock()
    data_module = MagicMock()
    cfg = OmegaConf.create({})

    # Mock the test results
    test_results = [{"test/accuracy": 0.95}]
    trainer.test.return_value = test_results

    # Mock save_benchmark_results to avoid file operations
    with patch("marsbench.training.execution.save_benchmark_results") as mock_save:
        _ = run_testing(trainer, model, data_module, cfg)

        # Verify trainer.test was called
        trainer.test.assert_called_once_with(model, data_module)

        # Verify results were saved
        mock_save.assert_called_once_with(cfg, test_results)


def test_run_prediction():
    """Test the prediction execution function."""
    # Create test objects
    trainer = MagicMock()
    model = MagicMock()
    data_module = MagicMock()
    cfg = OmegaConf.create({})

    # Mock prediction results
    predict_results = [{"preds": torch.tensor([1, 2, 3])}]
    trainer.predict.return_value = predict_results

    # Mock save_predictions to avoid file operations
    with patch("marsbench.training.execution.save_predictions") as mock_save:
        _ = run_prediction(trainer, model, data_module, cfg)

        # Verify trainer.predict was called
        trainer.predict.assert_called_once_with(model, data_module)

        # Verify results were saved
        mock_save.assert_called_once_with(cfg, predict_results)


def test_save_benchmark_results(tmp_path):
    """Test saving benchmark results."""
    # Create test directory structure
    benchmark_dir = os.path.join(tmp_path, "benchmarks")
    os.makedirs(benchmark_dir, exist_ok=True)

    # Create minimal config with required fields
    cfg = OmegaConf.create(
        {
            "model": {"name": "testmodel"},
            "data": {"name": "testdata", "num_classes": 10},
            "data_name": "testdata",
            "output_path": str(tmp_path),
        }
    )

    # Simple results
    results = [{"accuracy": 0.95}]

    # Mock pandas for CSV operations
    mock_df = MagicMock()
    mock_df_cls = MagicMock(return_value=mock_df)
    mock_concat = MagicMock(return_value=mock_df)

    # Mock various dependencies
    with patch("hydra.utils.get_original_cwd", return_value=str(tmp_path)), patch(
        "pandas.DataFrame", mock_df_cls
    ), patch("pandas.concat", mock_concat), patch("pandas.read_csv", side_effect=FileNotFoundError), patch(
        "os.path.exists", return_value=False
    ):

        # Run the function
        save_benchmark_results(cfg, results)

        # Verify DataFrame was created and to_csv was called
        mock_df_cls.assert_called()
        mock_df.to_csv.assert_called_once()


def test_save_predictions(tmp_path):
    """Test saving prediction outputs."""
    # Create minimal config with required fields
    cfg = OmegaConf.create(
        {"model": {"name": "testmodel"}, "data": {"name": "testdata"}, "prediction_output_path": str(tmp_path)}
    )

    # Simple predictions
    predictions = [{"preds": torch.tensor([1, 2, 3])}]

    # Mock path functions and file operations
    with patch("hydra.utils.get_original_cwd", return_value=str(tmp_path)), patch(
        "os.makedirs"
    ) as mock_makedirs, patch("numpy.savez_compressed") as _, patch("pandas.DataFrame.to_csv") as _:

        # Run the function
        save_predictions(cfg, predictions)

        # Verify directory was created
        mock_makedirs.assert_called()


def test_setup_model():
    """Test model setup."""
    # Create a comprehensive config with all the fields needed by the model
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "ResNet18",  # Use a model that exists in the registry
                "pretrained": False,
                "class_path": "marsbench.models.classification.ResNet18",
            },
            "data": {"name": "testdata", "num_classes": 10},
            "task": "classification",
        }
    )

    # Skip the actual model loading by patching at the right point
    with patch("marsbench.training.model_setup.import_model_class") as mock_import:
        # Create mock model
        mock_model = MagicMock(spec=pl.LightningModule)
        mock_model_class = MagicMock(return_value=mock_model)
        mock_import.return_value = mock_model_class

        # Run the function
        model = setup_model(cfg)

        # Verify model was created
        assert model is not None
        assert model == mock_model
        mock_model_class.assert_called_once_with(cfg)
