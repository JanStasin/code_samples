import sys
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory of the pytorch directory to the system path
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# # import the module
from preproc.data_prep import process_data
from models.nba_models.classifier import ClassifDs, ClassifNet
from src.run_class import class_torch_loop
from torch.utils.data import DataLoader


if __name__ == "__main__":
    pos = ['POINT GUARD', 'SHOOTING GUARD', 'SMALL FORWARD', 'POWER FORWARD', 'CENTER']
    chosen_feats = ['total_rebound_percentage', 'assist_percentage', 'steal_percentage','block_percentage', 
                    'three_point_attempt_rate', 'offensive_rebounds', 'blocks', 'assists', 'free_throw_attempt_rate','attempted_three_point_field_goals']

    X_scaled, Y = process_data('data/nba_stats_full_merged.csv', 'positions', pos, chosen_feats=chosen_feats)
    print(X_scaled.shape,  Y.shape)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, Y, test_size = 0.2)
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')

    NBA_data = ClassifDs(X_train=X_train, y_train=y_train)
    train_data = DataLoader(dataset=NBA_data, batch_size=32, shuffle=False) 

    n_feats = NBA_data.X.shape[1]## number of features in the dataset
    n_classes = len(NBA_data.y.unique()) ## get the unique labels

    ### Setting the hyper-parameters
    h_nodes = (20,10)
    n_epochs = 1000
    l = 0.00001 ## learning rate

    print( n_feats, h_nodes, n_classes, )

    model = ClassifNet(n_feats, h_nodes,  n_classes) ## create the model instance
    model.train()
   
    loss_func = torch.nn.CrossEntropyLoss()
    #cel = nn.BCEWithLogitsLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=l) ## define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=l)

    model, losses = class_torch_loop(train_data,
                                    n_epochs,
                                    optimizer,
                                    model,
                                    loss_func)

    print(len(losses))

    X_val_torch = torch.from_numpy(X_val)
    with torch.no_grad():
        y_val_hat_softmax = model(X_val_torch)
        y_val_hat = torch.max(y_val_hat_softmax.data, 1)

    ### Naive classifier accuracy
    most_common_cnt = Counter(y_val).most_common()[0][1]
    print(f"Naive Classifier: {most_common_cnt / len(y_val) * 100} %")

    test_acc = accuracy_score(y_val, y_val_hat.indices)
    print(f"Test accuracy: {test_acc * 100}%")

    plt.plot(losses[-int(np.round(len(losses)/32),0):])
    plt.show()



    