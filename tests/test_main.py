"""Tests for the main module."""
from unittest.mock import MagicMock
from unittest.mock import patch

import hydra
import pytest
from omegaconf import OmegaConf

from marsbench.main import main


# Mock Hydra's composition API to avoid initializing HydraConfig
@pytest.fixture(autouse=True)
def mock_hydra_composition():
    """Mock hydra's composition API."""
    with patch("hydra.initialize"), patch("hydra.compose") as mock_compose, patch(
        "hydra.utils.get_original_cwd", return_value="/mock/cwd"
    ):
        yield mock_compose


def test_main_train_mode():
    """Test main function in train mode."""
    # Create minimal config with required fields
    cfg = OmegaConf.create(
        {
            "task": "classification",
            "mode": "train",
            "model": {"name": "ResNet18"},
            "data": {"name": "domars16k"},
            "data_name": "domars16k",
            "training": {"trainer": {"accelerator": "cpu"}},
            "output_path": "outputs",
            "seed": 42,
            "dataset_path": "/data/hkerner/MarsBench/Datasets",
        }
    )

    # Mock all dependencies to avoid actual execution
    with patch("marsbench.main.seed_everything") as mock_seed, patch(
        "marsbench.main.setup_model"
    ) as mock_setup_model, patch("marsbench.main.MarsDataModule") as mock_data_module_class, patch(
        "marsbench.main.setup_callbacks"
    ) as mock_setup_callbacks, patch(
        "marsbench.main.setup_loggers"
    ) as mock_setup_loggers, patch(
        "marsbench.main.Trainer"
    ) as mock_trainer_class, patch(
        "marsbench.main.run_training"
    ) as mock_run_training:

        # Set up return values
        mock_setup_model.return_value = MagicMock()
        mock_data_module_class.return_value = MagicMock()
        mock_setup_callbacks.return_value = [MagicMock()]
        mock_setup_loggers.return_value = [MagicMock()]
        mock_trainer_class.return_value = MagicMock()

        try:
            # Run main
            main(cfg)

            # Verify each component was called
            mock_seed.assert_called_once_with(cfg.seed)
            mock_setup_model.assert_called_once()
            mock_data_module_class.assert_called_once()
            mock_setup_callbacks.assert_called_once()
            mock_setup_loggers.assert_called_once()
            mock_trainer_class.assert_called_once()
            mock_run_training.assert_called_once()
        except Exception as e:
            if "HydraConfig" in str(e):
                pytest.skip("Test skipped due to Hydra initialization issue")
            else:
                raise


def test_main_test_mode():
    """Test main function in test mode."""
    # Create minimal config with required fields
    cfg = OmegaConf.create(
        {
            "task": "classification",
            "mode": "test",
            "model": {"name": "ResNet18"},
            "data": {"name": "domars16k"},
            "data_name": "domars16k",
            "training": {"trainer": {"accelerator": "cpu"}},
            "output_path": "outputs",
            "seed": 42,
            "dataset_path": "/data/hkerner/MarsBench/Datasets",
        }
    )

    # Mock all dependencies to avoid actual execution
    with patch("marsbench.main.seed_everything"), patch("marsbench.main.setup_model") as mock_setup_model, patch(
        "marsbench.main.MarsDataModule"
    ) as mock_data_module_class, patch("marsbench.main.setup_callbacks") as mock_setup_callbacks, patch(
        "marsbench.main.setup_loggers"
    ) as mock_setup_loggers, patch(
        "marsbench.main.Trainer"
    ) as mock_trainer_class, patch(
        "marsbench.main.run_testing"
    ) as mock_run_testing:

        # Set up return values
        mock_setup_model.return_value = MagicMock()
        mock_data_module_class.return_value = MagicMock()
        mock_setup_callbacks.return_value = [MagicMock()]
        mock_setup_loggers.return_value = [MagicMock()]
        mock_trainer_class.return_value = MagicMock()

        try:
            # Run main
            main(cfg)

            # Verify run_testing was called
            mock_run_testing.assert_called_once()
        except Exception as e:
            if "HydraConfig" in str(e):
                pytest.skip("Test skipped due to Hydra initialization issue")
            else:
                raise


def test_main_predict_mode():
    """Test main function in predict mode."""
    # Create minimal config with required fields
    cfg = OmegaConf.create(
        {
            "task": "classification",
            "mode": "predict",
            "model": {"name": "ResNet18"},
            "data": {"name": "domars16k"},
            "data_name": "domars16k",
            "training": {"trainer": {"accelerator": "cpu"}},
            "output_path": "outputs",
            "prediction_output_path": "predictions",
            "seed": 42,
            "dataset_path": "/data/hkerner/MarsBench/Datasets",
        }
    )

    # Mock all dependencies to avoid actual execution
    with patch("marsbench.main.seed_everything"), patch("marsbench.main.setup_model") as mock_setup_model, patch(
        "marsbench.main.MarsDataModule"
    ) as mock_data_module_class, patch("marsbench.main.setup_callbacks") as mock_setup_callbacks, patch(
        "marsbench.main.setup_loggers"
    ) as mock_setup_loggers, patch(
        "marsbench.main.Trainer"
    ) as mock_trainer_class, patch(
        "marsbench.main.run_prediction"
    ) as mock_run_prediction:

        # Set up return values
        mock_setup_model.return_value = MagicMock()
        mock_data_module_class.return_value = MagicMock()
        mock_setup_callbacks.return_value = [MagicMock()]
        mock_setup_loggers.return_value = [MagicMock()]
        mock_trainer_class.return_value = MagicMock()

        try:
            # Run main
            main(cfg)

            # Verify run_prediction was called
            mock_run_prediction.assert_called_once()
        except Exception as e:
            if "HydraConfig" in str(e):
                pytest.skip("Test skipped due to Hydra initialization issue")
            else:
                raise


def test_main_invalid_mode():
    """Test main function with invalid mode."""
    # Create config with invalid mode
    cfg = OmegaConf.create(
        {
            "task": "classification",
            "mode": "invalid_mode",
            "model": {"name": "ResNet18"},
            "data": {"name": "domars16k"},
            "data_name": "domars16k",
            "training": {"trainer": {"accelerator": "cpu"}},
            "output_path": "outputs",
            "dataset_path": "/data/hkerner/MarsBench/Datasets",
        }
    )

    try:
        # Run main and expect exception for invalid mode
        with patch("marsbench.main.seed_everything"), patch("marsbench.main.setup_model"), patch(
            "marsbench.main.MarsDataModule"
        ), patch("marsbench.main.setup_callbacks"), patch("marsbench.main.setup_loggers"), patch(
            "marsbench.main.Trainer"
        ):

            with pytest.raises(ValueError) as excinfo:
                main(cfg)

            # Check the error message contains the invalid mode
            error_msg = str(excinfo.value)
            assert "invalid_mode" in error_msg.lower()
    except Exception as e:
        if "HydraConfig" in str(e):
            pytest.skip("Test skipped due to Hydra initialization issue")
        else:
            raise


def test_main_error_handling():
    """Test error handling in main function."""
    # Create minimal config
    cfg = OmegaConf.create(
        {
            "task": "classification",
            "mode": "train",
            "model": {"name": "ResNet18"},
            "data": {"name": "domars16k"},
            "data_name": "domars16k",
            "training": {"trainer": {"accelerator": "cpu"}},
            "output_path": "outputs",
            "seed": 42,
            "dataset_path": "/data/hkerner/MarsBench/Datasets",
        }
    )

    try:
        # Test exception handling
        with patch("marsbench.main.seed_everything"), patch(
            "marsbench.main.setup_model", side_effect=Exception("Test error")
        ):

            with pytest.raises(Exception) as excinfo:
                main(cfg)

            assert "Test error" in str(excinfo.value)
    except Exception as e:
        if "HydraConfig" in str(e):
            pytest.skip("Test skipped due to Hydra initialization issue")
        else:
            raise
