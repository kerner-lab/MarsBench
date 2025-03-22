# MarsBench

MarsBench is a comprehensive benchmarking framework for computer vision tasks on Mars surface images. It provides a unified platform for training, evaluating, and comparing machine learning models across various Mars datasets for classification, segmentation, and detection tasks.

## Features

- **Unified Interface**: Common framework for training and evaluating models across multiple datasets
- **Model Zoo**: Pre-configured implementations of popular vision models (ResNet, ViT, Swin Transformer, UNet, etc.)
- **Dataset Collection**: Standardized access to Mars image datasets with consistent APIs
- **Experiment Tracking**: Integration with WandB, TensorBoard, and CSV loggers
- **Flexible Configuration**: Hydra-based configuration system for experiment management
- **Reproducibility**: Automatic tracking of random seeds and experiment configurations
- **Extensibility**: Easy addition of new models and datasets

## Installation

```bash
# Install the package with core dependencies
pip install -e .

# Install with development dependencies (for testing, linting, etc.)
pip install -e ".[dev]"
```

## Quick Start

To train a model:

```bash
python -m marsbench.main task=classification model_name=resnet18 data_name=domars16k data_path=/path/to/dataset
```

To test a model (after training):

```bash
python -m marsbench.main mode=test task=classification model_name=resnet18 data_name=domars16k checkpoint_path=outputs/classification/domars16k/resnet18/YYYY-MM-DD_HH-MM-SS/checkpoints/best.ckpt
```

To generate predictions:

```bash
python -m marsbench.main mode=predict task=classification model_name=resnet18 data_name=domars16k checkpoint_path=outputs/classification/domars16k/resnet18/YYYY-MM-DD_HH-MM-SS/checkpoints/best.ckpt
```

## Configuration

MarsBench uses Hydra for configuration management. The main configuration is located in `configs/config.yaml` with task-specific configurations in the respective subdirectories:

- `configs/model/classification/`: Classification model configurations
- `configs/model/segmentation/`: Segmentation model configurations
- `configs/model/detection/`: Detection model configurations
- `configs/data/classification/`: Classification dataset configurations
- `configs/data/segmentation/`: Segmentation dataset configurations
- `configs/data/detection/`: Detection dataset configurations
- `configs/training/`: Training hyperparameters and settings
- `configs/transforms/`: Image transformation settings
- `configs/logger/`: Logging configuration
- `configs/callbacks/`: PyTorch Lightning callback settings

Override any configuration parameter using the command line:

```bash
python -m marsbench.main task=classification model_name=resnet18 data_name=domars16k training.batch_size=64 training.optimizer.lr=0.0005
```

## Supported Datasets

### Classification
- **DoMars16k**: Mars terrain classification (15 classes)
- **MSLNet**: Mars Science Laboratory image classification (19 classes)
- **HiRISENet**: High-Resolution Imaging Science Experiment images
- **DeepMars_Surface**: Mars surface type classification
- **DeepMars_Landmark**: Mars landmark classification
- **MartianFrost**: Mars frost detection

### Segmentation
- **ConeQuest**: Segmentation of cone-like structures on Mars surface

### Detection
- To be added

## Supported Models

### Classification
- ResNet18, ResNet50
- Vision Transformer (ViT)
- Swin Transformer
- InceptionV3
- SqueezeNet

### Segmentation
- UNet
- DeepLab

### Detection
- To be added

## Project Structure

```
marsbench/
├── configs/                # Hydra configuration files
├── marsbench/              # Main package
│   ├── data/               # Dataset implementations
│   │   ├── classification/ # Classification datasets
│   │   └── segmentation/   # Segmentation datasets
│   ├── models/             # Model implementations
│   │   ├── classification/ # Classification models
│   │   └── segmentation/   # Segmentation models
│   ├── training/           # Training utilities
│   ├── utils/              # Helper functions
│   └── main.py             # Entry point
├── tests/                  # Unit tests
├── examples/               # Example scripts
└── outputs/                # Generated outputs (predictions, checkpoints)
```

## Development

### Adding a New Dataset

1. Create a new dataset implementation in `marsbench/data/classification/` or `marsbench/data/segmentation/`
2. Inherit from `BaseClassificationDataset` or `BaseSegmentationDataset`
3. Implement the `_load_data` method
4. Add dataset configuration in `configs/data/classification/` or `configs/data/segmentation/`
5. Register the dataset in `marsbench/data/__init__.py`

### Adding a New Model

1. Create a new model implementation in `marsbench/models/classification/` or `marsbench/models/segmentation/`
2. Inherit from `BaseClassificationModel` or `BaseSegmentationModel`
3. Implement the `_initialize_model` method
4. Add model configuration in `configs/model/classification/` or `configs/model/segmentation/`
5. Register the model in `marsbench/models/__init__.py`

## Testing

Run the test suite:

```bash
pytest tests/
```
