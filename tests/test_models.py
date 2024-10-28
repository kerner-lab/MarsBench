import os
import glob
import pytest
import torch
import importlib
import tempfile
from hydra import initialize_config_dir, compose
from src.models import *
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer


@pytest.mark.parametrize("model_config_file", glob.glob("configs/model/*.yaml"))
def test_models(model_config_file):
    model = os.path.splitext(os.path.basename(model_config_file))[0]
    config_dir = os.path.abspath('configs')

    # Initialize Hydra and compose configuration
    with initialize_config_dir(config_dir=config_dir, version_base='1.1'):
        cfg = compose(
            config_name='config',
            overrides=[
                f'model={model}',
                'data.num_classes=10',
            ],
        )

    for objective in cfg.model:
        model_name = cfg.model.get(objective).name
        # Check model status
        if cfg.model.get(objective).status != 'ready':
            print(f"Skipping model '{model_name}' for {objective} (status: {cfg.model.get(objective).status})")
            pytest.skip(f"Model '{model_name}' for {objective} is not ready for testing.")

        print(f"Testing model '{model_name}' for {objective}")
        model_class_path = cfg.model.get(objective, {}).get('class_path', None)
        if model_class_path is None:
            pytest.fail(f"Model class path not specified for model '{model_name}' in the configuration.")

        module_name, class_name = model_class_path.rsplit('.', 1)
        try:
            module = importlib.import_module(module_name)
            ModelClass = getattr(getattr(module, class_name), class_name)
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Failed to import model class '{model_class_path}' for model '{model_name}': {e}")
        input_size = cfg.model.get(objective).get('input_size', [3, 224, 224])
        model = ModelClass(cfg)
        batch_size = 2
        dummy_input = torch.randn(batch_size, *input_size, requires_grad=True)
        dummy_target = torch.randint(0, cfg.data.num_classes, (batch_size,))
        
        # Perform forward pass
        model.train()
        output = model(dummy_input)
        
        expected_output_shape = (batch_size, cfg.data.num_classes)
        if isinstance(output, tuple) and model_name in cfg.test.model.with_tuple_output:
            output = output[0]
        elif isinstance(output, tuple):
            pytest.fail(f"Not expecting tuple as output for Model: '{model_name}'.")

        assert output.shape == expected_output_shape, \
            f"{model_name}: Expected output shape {expected_output_shape}, got {output.shape}"

        print(f"{model_name}: Forward pass successful with output shape {output.shape}")

        # Perform backward pass
        criterion_name = cfg.criterion.name
        if criterion_name == 'cross_entropy':
            criterion = torch.nn.CrossEntropyLoss()
        else:
            pytest.fail(f"Criterion '{criterion_name}' not recognized.")

        loss = criterion(output, dummy_target)
        loss.backward()

        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        assert grad_norm > 0, f"{model_name}: Gradients are not computed during backward pass."

        # Perform a training step
        class DummyDataset(Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                dummy_input = torch.randn(*input_size)
                dummy_label = torch.randint(0, cfg.data.num_classes, (1,)).item()
                return dummy_input, dummy_label

        train_dataloader = DataLoader(DummyDataset(), batch_size=cfg.training.batch_size)
        val_dataloder = DataLoader(DummyDataset(), batch_size=cfg.training.batch_size)

        trainer = Trainer(max_epochs=cfg.training.max_epochs, fast_dev_run=True)

        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloder)

        print(f"{model_name}: Training integration test successful")

        # Save model
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(model.state_dict(), tmp.name)
            model_loaded = ModelClass(cfg)
            model_loaded.load_state_dict(torch.load(tmp.name, weights_only=False))

        for param_original, param_loaded in zip(model.parameters(), model_loaded.parameters()):
            assert torch.allclose(param_original, param_loaded), \
                f"{model_name}: Parameters differ after loading."

        print(f"{model_name}: Model saving and loading successful")