import torch
import torch.nn as nn
import numpy as np


def class_torch_loop(train_data, n_epochs, optimizer, model, loss_func, get_losses=True):
    """
    Trains a PyTorch model using a loop over the training data.

    Args:
        train_data (iterable): The training data, typically a DataLoader or a custom iterable.
        n_epochs (int): The number of epochs to train the model for.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.
        model (torch.nn.Module): The PyTorch model to be trained.
        loss_func (callable): The loss function used to calculate the loss between predictions and targets.
        get_losses (bool, optional): Whether to collect and return the losses during training. Defaults to True.

    Returns:
        tuple: A tuple containing the trained model and a list of losses (if get_losses is True).
    """
    losses = []
    for e in range(n_epochs):
        for X, y in train_data:
            # set gradients to 0:
            optimizer.zero_grad()
            # forward -->
            y_pred = model(X)
            # loss calculation
            loss = loss_func(y_pred, y)
            if get_losses:
                losses.append(loss.item()) 
            # backward <--
            loss.backward()
            # update the weights
            optimizer.step()
        if e % (np.round(n_epochs/10,0)) == 0: 
            print(f"Epoch {e}/{n_epochs}, Loss: {loss.data:2f}")
    return model, losses