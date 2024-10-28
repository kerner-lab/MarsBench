import os
import glob
import pytest
from hydra import initialize_config_dir, compose
from src.utils.transforms import get_transforms
from src.data import get_dataset

@pytest.mark.parametrize("dataset_config_file", glob.glob("configs/data/*.yaml"))
def test_datasets(dataset_config_file):
    # Get dataset name from config file name
    dataset_name = os.path.splitext(os.path.basename(dataset_config_file))[0]
    config_dir = os.path.abspath('configs')

    # Initialize Hydra and compose configuration
    with initialize_config_dir(config_dir=config_dir, version_base='1.1'):
        cfg = compose(config_name='config', overrides=[f'data={dataset_name}'])

        # Check dataset status
        if cfg.data.status not in cfg.test.data.status:
            print(f"Skipping dataset '{dataset_name}' (status: {cfg.data.status})")
            pytest.skip(f"Dataset '{dataset_name}' is not in status set {cfg.test.data.status} for testing.")

        print(f"Testing dataset '{dataset_name}'")

        # Get transforms
        train_transform, val_transform = get_transforms(cfg)

        # Get datasets
        train_dataset, val_dataset, test_dataset = get_dataset(cfg, train_transform, val_transform, subset=cfg.test.data.subset_size)

        # Expected image size and channels
        expected_image_size = cfg.transforms.image_size
        expected_channels = 3  # Assuming images are in RGB format

        # Function to test a dataset split
        def test_split(split_name, dataset):
            if len(dataset) == 0:
                print(f"The {split_name} dataset of '{dataset_name}' is empty.")
                pytest.fail(f"The {split_name} dataset of '{dataset_name}' is empty.")

            # Test a few samples
            sample_indices = [0, len(dataset) // 2, len(dataset) - 1]
            for idx in sample_indices:
                image, label = dataset[idx]

                # Check image shape
                assert image.shape[0] == expected_channels, (
                    f"Dataset '{dataset_name}' {split_name} split: Expected {expected_channels} channels, "
                    f"got {image.shape[0]}"
                )
                assert image.shape[1] == expected_image_size[0], (
                    f"Dataset '{dataset_name}' {split_name} split: Expected height {expected_image_size[0]}, "
                    f"got {image.shape[1]}"
                )
                assert image.shape[2] == expected_image_size[1], (
                    f"Dataset '{dataset_name}' {split_name} split: Expected width {expected_image_size[1]}, "
                    f"got {image.shape[2]}"
                )

                # Check label
                assert isinstance(label, int), (
                    f"Dataset '{dataset_name}' {split_name} split: Label is not an integer."
                )
                num_classes = cfg.data.num_classes
                assert 0 <= label < num_classes, (
                    f"Dataset '{dataset_name}' {split_name} split: Label {label} out of range [0, {num_classes - 1}]."
                )

                print(
                    f"Dataset '{dataset_name}' {split_name} split, index {idx}: "
                    f"Image shape {image.shape}, Label {label}"
                )

        # Test each split
        test_split('train', train_dataset)
        test_split('val', val_dataset)
        test_split('test', test_dataset)
