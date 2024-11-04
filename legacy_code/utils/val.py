import numpy as np
import torch
import wandb
from sklearn.metrics import precision_recall_fscore_support


def validate(model, val_loader, device):
    all_preds = []
    all_labels = []
    # Set the model to evaluation mode
    model.eval()

    # Iterate over the batches of the validation loader
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Append current batch predictions and labels to lists
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate precision, recall, F1 score
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )

    return precision, recall, f1_score


def test(model, test_loader, device, test_len, criterion):
    running_loss = 0.0
    running_corrects = 0
    # Iterating over the batches of the test loader
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Update the running loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    # Calculate the validation loss and accuracy
    test_loss = running_loss / test_len
    test_acc = running_corrects.double() / test_len

    metrics = {"test_loss": test_loss, "test_acc": test_acc}

    return metrics
