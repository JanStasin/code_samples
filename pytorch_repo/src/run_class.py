import torch
import torch.nn as nn
import numpy as np


def class_torch_loop(train_data, n_epochs, optimizer, model, loss_func, get_losses=True):
    losses = []
    for e in range(n_epochs):
        for X, y in train_data:
            #set gradients to 0:
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