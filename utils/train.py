import torch
import wandb
import psutil
from utils.val import validate



def train(model, train_loader, val_loader, criterion, optimizer, device, train_len, val_len, num_epochs):
  # Train the model for the specified number of epochs
  mem_info = psutil.virtual_memory()
  for epoch in range(num_epochs):
  # Set the model to train mode
    model.train()

    # Initialize the running loss and accuracy
    running_loss = 0.0
    running_corrects = 0
    wandb.log({"Memory Usage (MB)": mem_info.used / (1024 ** 2)}, step=epoch)
    # Iterate over the batches of the train loader
    for inputs, labels in train_loader:
        # Move the inputs and labels to the device
      inputs = inputs.to(device)
      labels = labels.to(device)
      # print(labels)

      # Zero the optimizer gradients
      optimizer.zero_grad()

      # Forward pass
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      loss = criterion(outputs, labels)

      # Backward pass and optimizer step
      loss.backward()
      optimizer.step()
      metric1 = {'train_step_loss':loss}
      wandb.log(metric1)
      # Update the running loss and accuracy
      running_loss += loss.item() * inputs.size(0)
      metric2={'train_running_loss':running_loss}
      running_corrects += torch.sum(preds == labels.data)

    # Calculate the train loss and accuracy

    train_loss = running_loss / train_len
    train_acc = running_corrects.double() / train_len

    # Set the model to evaluation mode
    model.eval()

    # Initialize the running loss and accuracy
    running_loss = 0.0
    running_corrects = 0

    # Iterate over the batches of the validation loader
    with torch.no_grad():
      for inputs, labels in val_loader:
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
    val_loss = running_loss / val_len
    val_acc = running_corrects.double() / val_len
    metrics3 = {"training_loss":train_loss,'validation_loss':val_loss}
    metrics4 = {"training_acc":train_acc,'validation_acc':val_acc}
    wandb.log(metrics3)
    wandb.log(metrics4)
    # Print the epoch results
    print('Epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'
          .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))
    validate(model, val_loader, device)

