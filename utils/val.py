
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import wandb




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
  precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
  metrics4={"precision":precision,"recall":recall,"f1_score":f1_score}
  wandb.log(metrics4)
  # Print the epoch results
  print('val precision: {:.4f}, val recall: {:.4f}, val F1 score: {:.4f}'
        .format(precision, recall, f1_score))