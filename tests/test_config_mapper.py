"""
Tests for the config_mapper module.
"""

import os
import sys
from pathlib import Path

import pytest
from hydra import compose
from hydra import initialize_config_dir
from omegaconf import OmegaConf

# Import the config_mapper functions
from marsbench.utils.config_mapper import ConfigLoadError
from marsbench.utils.config_mapper import load_dynamic_configs


@pytest.fixture
def base_config():
    """Create a basic configuration for testing."""
    return OmegaConf.create(
        {
            "task": "classification",
            "data_name": "test_dataset",
            "model_name": "test_model",
        }
    )


@pytest.fixture
def mock_config_dir(tmp_path):
    """Create a temporary config directory with test configurations."""
    # Create config directories
    config_dir = tmp_path / "configs"
    data_dir = config_dir / "data"
    model_dir = config_dir / "model"

    # Create task-specific directories
    cls_data_dir = data_dir / "classification"
    cls_model_dir = model_dir / "classification"
    seg_data_dir = data_dir / "segmentation"
    seg_model_dir = model_dir / "segmentation"

    # Create all directories with parents
    for directory in [cls_data_dir, cls_model_dir, seg_data_dir, seg_model_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Create test config files
    classification_data_config = OmegaConf.create({"name": "TestDataset", "status": "test", "num_classes": 5})

    classification_model_config = OmegaConf.create(
        {
            "name": "TestModel",
            "class_path": "marsbench.models.test.TestModel",
            "input_size": [3, 224, 224],
        }
    )

    segmentation_data_config = OmegaConf.create({"name": "SegDataset", "task": "segmentation", "num_classes": 2})

    segmentation_model_config = OmegaConf.create({"name": "SegModel", "encoder_name": "resnet34"})

    # Write config files
    OmegaConf.save(classification_data_config, cls_data_dir / "test_dataset.yaml")
    OmegaConf.save(classification_model_config, cls_model_dir / "test_model.yaml")
    OmegaConf.save(segmentation_data_config, seg_data_dir / "test_dataset.yaml")
    OmegaConf.save(segmentation_model_config, seg_model_dir / "test_model.yaml")

    return tmp_path


class TestConfigMapper:
    """Test the config_mapper module functionality."""

    def test_load_dynamic_configs(self, base_config, mock_config_dir):
        """Test that dynamic configs are loaded correctly."""
        # Load configs from the mock directory
        result = load_dynamic_configs(base_config, config_dir=mock_config_dir / "configs")

        # Verify data config was loaded
        assert hasattr(result, "data")
        assert "testdataset" in str(result.data).lower()
        assert "test" in str(result.data).lower()
        assert result.data.num_classes == 5

        # Verify model config was loaded
        assert hasattr(result, "model")
        assert "testmodel" in str(result.model).lower()
        assert "marsbench.models.test.testmodel" in str(result.model).lower()
        assert result.model.input_size == [3, 224, 224]

    def test_load_dynamic_configs_with_nonexistent_data(self, base_config, mock_config_dir):
        """Test loading configs with nonexistent data file."""
        # Modify config to use nonexistent data
        config = OmegaConf.create(OmegaConf.to_container(base_config))
        config.data_name = "nonexistent_dataset"

        # In non-test environment, this should raise an error
        with pytest.raises(ConfigLoadError):
            load_dynamic_configs(config, config_dir=mock_config_dir / "configs")

        # In test environment, it should continue without error
        os.environ["TEST_ENV"] = "1"
        try:
            result = load_dynamic_configs(config, config_dir=mock_config_dir / "configs")
            assert not hasattr(result, "data") or result.data is None
        finally:
            if "TEST_ENV" in os.environ:
                del os.environ["TEST_ENV"]

        # Verify model config WAS loaded
        assert hasattr(result, "model")
        assert "testmodel" in str(result.model).lower()

    def test_integration_with_task_change(self, base_config, mock_config_dir):
        """Test that changing task affects the paths used for configs."""
        # Create segmentation config
        config = OmegaConf.create(OmegaConf.to_container(base_config))
        config.task = "segmentation"

        # Load dynamic configs
        result = load_dynamic_configs(config, config_dir=mock_config_dir / "configs")

        # Verify correct segmentation configs were loaded
        assert hasattr(result, "data")
        assert "segdataset" in str(result.data).lower()
        assert "segmentation" in str(result.data).lower()

        assert hasattr(result, "model")
        assert "segmodel" in str(result.model).lower()
        assert "resnet34" in str(result.model).lower()


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "configs")),
    reason="Project config files not available",
)
class TestIntegrationWithProjectConfigs:
    """Integration tests with the actual project configurations."""

    def test_basic_config_loading(self):
        """Test basic configuration loading functionality."""
        # Get the project config directory
        project_root = Path(__file__).parent.parent
        config_dir = project_root / "configs"

        # Use Hydra's compose to load the base config
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(
                config_name="config",
                overrides=[
                    "task=classification",
                    "data_name=domars16k",
                    "model_name=resnet18",
                ],
            )

            # Apply dynamic config loading
            result = load_dynamic_configs(cfg)

        # Verify configs were loaded
        assert hasattr(result, "data"), "Data config not loaded"
        assert hasattr(result, "model"), "Model config not loaded"

        # Verify core attributes
        assert hasattr(result.data, "num_classes"), "Data config missing num_classes"
        assert hasattr(result.model, "name"), "Model config missing name attribute"

    def test_dynamic_config_variations(self):
        """
        Test different variations of dynamic config loading.

        This test documents the expected behavior of load_dynamic_configs
        when handling different model and task configurations.

        Once the implementation of load_dynamic_configs is fixed,
        uncomment the code below to validate the correct behavior.
        """
        # Get project config directory
        project_root = Path(__file__).parent.parent
        config_dir = project_root / "configs"

        # Test case 1: Classification with Default ResNet
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            # Load classification config
            cfg_resnet = compose(
                config_name="config",
                overrides=[
                    "task=classification",
                    "data_name=domars16k",
                    "model_name=resnet18",
                ],
            )

            # Load dynamic configs
            result_resnet = load_dynamic_configs(cfg_resnet)

            # Verify classification model configs were loaded correctly
            assert hasattr(result_resnet, "model"), "Classification model config not loaded"
            assert result_resnet.model.name.lower() == "resnet18", "Expected ResNet model"

        # Test case 2: Segmentation with UNet
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            # Load segmentation config
            cfg_unet = compose(
                config_name="config",
                overrides=[
                    "task=segmentation",
                    "data_name=cone_quest",
                    "model_name=unet",
                ],
            )

            # Load dynamic configs
            result_unet = load_dynamic_configs(cfg_unet)
            print(result_unet)
            # Verify segmentation model configs were loaded correctly
            assert hasattr(result_unet, "model"), "Segmentation model config not loaded"
            assert result_unet.model.name.lower() == "unet", "Expected UNet model"
            assert result_unet.task == "segmentation", "Wrong task type for segmentation"

        # Test case 3: Different model for same task
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            # Load with a different classification model
            cfg_alt = compose(
                config_name="config",
                overrides=[
                    "task=classification",
                    "data_name=domars16k",
                    "model_name=vit",  # Different model
                ],
            )

            # Load dynamic configs
            result_alt = load_dynamic_configs(cfg_alt)

            # Verify different model configs are loaded for different model_name
            assert hasattr(result_alt, "model"), "Alternative model config not loaded"
            # Model names should be different for different model_name values
            assert (
                result_alt.model.name.lower() != result_resnet.model.name.lower()
            ), "Different models should have different names"

    def test_config_overrides(self):
        """Test that overrides are preserved during dynamic loading."""
        project_root = Path(__file__).parent.parent
        config_dir = project_root / "configs"

        # Create config with Hydra compose and add overrides
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(
                config_name="config",
                overrides=[
                    "task=classification",
                    "data_name=domars16k",
                    "model_name=resnet18",
                    "training.batch_size=64",  # Custom override
                    "training.early_stopping_patience=10",  # Custom override
                ],
            )

            # Apply dynamic config loading
            result = load_dynamic_configs(cfg)

        # Verify overrides are preserved
        assert result.training.batch_size == 64, "Custom batch_size override was lost"
        assert result.training.early_stopping_patience == 10, "Custom patience override was lost"

        # Verify model was still loaded
        assert hasattr(result, "model"), "Model config not loaded with overrides present"

    def test_error_handling(self):
        """Test error handling for invalid configurations."""
        project_root = Path(__file__).parent.parent
        config_dir = project_root / "configs"

        # Test with invalid task
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            invalid_task_cfg = compose(
                config_name="config",
                overrides=[
                    "task=nonexistent_task",  # Invalid task
                    "data_name=domars16k",
                    "model_name=resnet18",
                ],
            )

        # With TEST_ENV=1, should not raise errors
        os.environ["TEST_ENV"] = "1"
        try:
            _ = load_dynamic_configs(invalid_task_cfg, config_dir=config_dir)
            # Test mode should prevent exceptions
            assert True, "Config loading should not fail in TEST_ENV"
        finally:
            if "TEST_ENV" in os.environ:
                del os.environ["TEST_ENV"]
