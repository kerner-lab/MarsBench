import os
import pytest
import torch
from hydra import initialize_config_dir, compose
from src.data.mars_datamodule import MarsDataModule

def test_mars_datamodule():
    # Load a sample configuration
    config_dir = os.path.abspath('configs')
    with initialize_config_dir(config_dir=config_dir, version_base='1.1'):
        cfg = compose(config_name='config')
    # Initialize the MarsDataModule
    data_module = MarsDataModule(cfg)

    # Call setup to load datasets
    data_module.setup()

    # Check that the datasets are loaded
    assert data_module.train_dataset is not None, "train_dataset is not loaded."
    assert data_module.val_dataset is not None, "val_dataset is not loaded."
    assert data_module.test_dataset is not None, "test_dataset is not loaded."

    # Create data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Fetch a single batch from each DataLoader
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))

    # Verify the batch content (images and labels)
    train_images, train_labels = train_batch
    val_images, val_labels = val_batch
    test_images, test_labels = test_batch

    # Check shapes and types
    assert train_images.shape[1:] == (3, cfg.transforms.image_size[0], cfg.transforms.image_size[1]), \
        f"Train images have incorrect shape: {train_images.shape}"
    assert val_images.shape[1:] == (3, cfg.transforms.image_size[0], cfg.transforms.image_size[1]), \
        f"Validation images have incorrect shape: {val_images.shape}"
    assert test_images.shape[1:] == (3, cfg.transforms.image_size[0], cfg.transforms.image_size[1]), \
        f"Test images have incorrect shape: {test_images.shape}"

    assert train_labels.dtype == torch.int64, "Train labels are not integers."
    assert val_labels.dtype == torch.int64, "Validation labels are not integers."
    assert test_labels.dtype == torch.int64, "Test labels are not integers."

    print("MarsDataModule datasets and DataLoaders are working correctly.")

if __name__ == "__main__":
    pytest.main([__file__])
