"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
  """Trains a PyTorch model for a single epoch.
  Turns a target PyTorch model to training mode and then runs through all the 
  required training steps (forward pass, loss calculation, optimizer step).
  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").
  Returns:  
    A tuple containing the training loss and training accuracy metrics.
    In the form of (train_loss, train_accuracy). 
    For example:
    (0.1112, 0.8743)
  """
  
  # Put model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  # Loop through data loader and data batches
  for batch, (X_train, y_train) in enumerate(dataloader):
    
    # Send data to target device
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    # 1. Perform the forward pass
    y_pred_logits_train = model(X_train)

    # 2. Calculate and accumulate loss
    loss_this_batch = loss_fn(y_pred_logits_train, y_train)
    train_loss += loss_this_batch.item()

    # 3. Zero the gradients
    optimizer.zero_grad()

    # 4. Backpropagation
    loss_this_batch.backward()

    # 5. Step through and update the weights
    optimizer.step()

    # Calculate and accumilate accuracy metric across all batches
    y_pred_labels_train = torch.argmax(torch.softmax(y_pred_logits_train, dim = 1), dim = 1)
    train_acc += (y_pred_labels_train == y_train).sum().item() / len(y_pred_logits_train)

  # Adjust the metrics to get average loss and accuracy per batch
  train_loss /= len(dataloader)
  train_acc /= len(dataloader)
  return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

  """Tests a PyTorch model for a single epoch.
  Turns a target PyTorch model to "eval" mode and then performs a forward pass on the testing dataset.
  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch model to be tested.
    device: A target deice to compute on (e.g. "cuda" or "cpu").
  Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). 
    For example:
    (0.0223, 0.8985)
  """

  # Put model in eval mode
  model.eval()

  # Set up test loss and test accuracy values
  test_loss, test_acc = 0, 0

  # Turn on inference mode context manager
  with torch.inference_mode():
    # Loop through the DataLoader batches
    for batch, (X_test, y_test) in enumerate(dataloader):
      # Send the data to the target device
      X_test = X_test.to(device)
      y_test = y_test.to(device)

      # 1. Perform the forward pass
      y_pred_logits_test = model(X_test)

      # 2. Calculate and accumulate loss
      loss_this_batch = loss_fn(y_pred_logits_test, y_test)
      test_loss += loss_this_batch.item()

      # 3. Calculate and accumulate accuracy
      y_pred_labels_test = y_pred_logits_test.argmax(dim = 1)
      test_acc += (y_pred_labels_test == y_test).sum().item() / len(y_pred_labels_test)
    
    # Adjust metrics to get average loss and accuracy per batch
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device, 
          writer: torch.utils.tensorboard.writer.SummaryWriter = None # new parameter to take in a writer
          ) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Stores metrics to specified writer log_dir if present.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      test_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").
      writer: A SummaryWriter() instance to log model results to.

    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for 
      each epoch.
      In the form: {train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...]} 
      For example if training for epochs=2: 
              {train_loss: [2.0616, 1.0537],
                train_acc: [0.3945, 0.3945],
                test_loss: [1.2641, 1.5706],
                test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)


        # Use the writer parameter to track experiments 
        # See if there's a writer, if so, log to it
        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(main_tag="Loss", 
                               tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag="Accuracy", 
                               tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc}, 
                               global_step=epoch)

            # Close the writer
            writer.close()

    # Return the filled results at the end of the epochs
    return results