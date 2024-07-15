import sys
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Add the parent directory of the pytorch directory to the system path
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.classifier import ClassifDs, ClassifNet
from src.run_class import class_torch_loop
from torch.utils.data import DataLoader

def hyperparameter_tuning(X,Y, param_grid):
    """
    Perform hyperparameter tuning using grid search.
    param_grid = {
        'n_epochs': [100, 500, 1000, 1500],
        'h_nodes': [(10,10),(20, 10), (30, 15)],
        'l': [0.001, 0.0005, 0.00001]
    }
    """
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size = 0.2)
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')

    DATA = ClassifDs(X_train=X_train, y_train=y_train)
    train_data = DataLoader(dataset=DATA, batch_size=32, shuffle=False) 

    n_feats = X_train.shape[1]
    n_classes = len(np.unique(Y)) 

    # Initialize the model
    model = ClassifNet(n_feats=n_feats, h_nodes=(4, 4), n_classes=n_classes)
    model.train()

    # Define the loss function and optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Placeholder for storing the best parameters and corresponding score
    best_params = None
    best_score = -np.inf

    # Iterate over the hyperparameter grid
    for params in param_grid['n_epochs']:
        for nodes in param_grid['h_nodes']:
            for learning_rate in param_grid['l']:
                # Update the model with new hyperparameters
                model = ClassifNet(n_feats=n_feats, h_nodes=nodes, n_classes=n_classes)
                model.train()

                # Update the optimizer with the new learning rate
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                # Train the model
                _, losses = class_torch_loop(train_data, params, optimizer, model, loss_func)

                # Evaluate the model on the validation set
                with torch.no_grad():
                    y_val_hat_softmax = model(torch.from_numpy(X_val))
                    y_val_hat = torch.max(y_val_hat_softmax.data, 1)
                    val_accuracy = accuracy_score(y_val, y_val_hat.indices)

                # Update the best parameters and score if necessary
                if val_accuracy > best_score:
                    best_score = val_accuracy
                    best_params = {'n_epochs': params, 'h_nodes': nodes, 'l': learning_rate}

    return best_params, best_score

