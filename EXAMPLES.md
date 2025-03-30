# MarsBench Usage Examples

This document provides ready-to-use examples for running MarsBench commands in various scenarios.

## Basic Commands

### Training a Model

```bash
# Basic classification training
python -m marsbench.main task=classification model_name=resnet18 data_name=domars16k

# Segmentation training with specific data path
python -m marsbench.main task=segmentation model_name=unet data_name=cone_quest dataset_path=/path/to/data

# Training with test after training
python -m marsbench.main task=classification model_name=resnet50 data_name=msl_net test_after_training=true
```

### Testing a Trained Model

```bash
# Test a model
python -m marsbench.main mode=test task=classification model_name=resnet18 data_name=domars16k

# Test a trained model using its checkpoint
python -m marsbench.main mode=test task=classification model_name=resnet18 data_name=domars16k checkpoint_path=outputs/classification/domars16k/resnet18/YYYY-MM-DD_HH-MM-SS/checkpoints/best.ckpt

# Test a model and save benchmark results with custom output path
python -m marsbench.main mode=test task=classification model_name=resnet18 data_name=domars16k checkpoint_path=outputs/classification/domars16k/resnet18/YYYY-MM-DD_HH-MM-SS/checkpoints/best.ckpt output_path=results/my_benchmark
```

### Generating Predictions

```bash
# Inference
python -m marsbench.main mode=predict task=classification model_name=resnet18 data_name=domars16k

# Generate predictions with a trained model
python -m marsbench.main mode=predict task=classification model_name=resnet18 data_name=domars16k checkpoint_path=outputs/classification/domars16k/resnet18/YYYY-MM-DD_HH-MM-SS/checkpoints/best.ckpt

# Predictions with custom output path
python -m marsbench.main mode=predict task=classification model_name=resnet18 data_name=hirise_net checkpoint_path=outputs/classification/hirise_net/resnet18/latest/checkpoints/best.ckpt prediction_output_path=predictions/hirise_results
```

## Advanced Usage

### Customizing Training Parameters

```bash
# Change learning rate and batch size
python -m marsbench.main task=classification model_name=resnet18 data_name=domars16k training.optimizer.lr=0.001 training.batch_size=64

# Use a different optimizer
python -m marsbench.main task=classification model_name=resnet18 data_name=domars16k training.optimizer.name=AdamW

# Set the number of epochs
python -m marsbench.main task=classification model_name=resnet18 data_name=domars16k training.max_epochs=50
```

### Data and Transforms

```bash
# Apply custom data transformations
python -m marsbench.main task=classification model_name=resnet18 data_name=domars16k transforms=default

# Specify a different validation split
python -m marsbench.main task=classification model_name=resnet18 data_name=domars16k training.trainer.limit_val_batches=0.2

# Use a subset of data for quick experiments (add + before the key for default undefineds)
python -m marsbench.main task=classification model_name=resnet18 data_name=domars16k +data.subset=1000
```

### Logging and Callbacks

```bash
# Enable WandB logging
python -m marsbench.main task=classification model_name=resnet18 data_name=domars16k logger.wandb.enabled=true

# Set early stopping parameters (alternately set early stop patience training.early_stopping_patience)
python -m marsbench.main task=classification model_name=resnet18 data_name=domars16k callbacks.early_stopping.patience=10 callbacks.early_stopping.monitor=val/accuracy callbacks.early_stopping.mode=max

# Save multiple checkpoints
python -m marsbench.main task=classification model_name=resnet18 data_name=domars16k callbacks.best_checkpoint.save_top_k=3
```
