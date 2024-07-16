### dataset and model class for nba data analysis
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ClassifDs(Dataset):
    def __init__(self, X_train, y_train):
        super().__init__()
         ## create the respective tensors for X and y
        self.X = torch.from_numpy(X_train) 
        self.y = torch.from_numpy(y_train)
        self.y = self.y.type(torch.LongTensor)
        self.len = self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return self.len

class MultiLabelDs(Dataset):
    def __init__(self, X_train, y_train):
        super().__init__()
         ## create the respective tensors for X and y
        self.X = X_train
        self.y = y_train
        self.y = self.y.type(torch.LongTensor)
        self.len = self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return self.len

class ClassifNet(nn.Module):
    def __init__(self, n_feats, h_nodes,  n_classes):
        super().__init__()
        self.lin1 = nn.Linear(n_feats, h_nodes[0])
        self.lin2 = nn.Linear(h_nodes[0], h_nodes[1])
        self.lin3 = nn.Linear(h_nodes[1], n_classes)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.log_softmax(x)
        return x

class MLClassifNet(nn.Module):
    def __init__(self, n_feats,  h_feats,  n_classes):
        #super().__init__()
        super(MLClassifNet, self).__init__()
        self.lin1 = nn.Linear(n_feats, h_feats[0])
        self.lin2 = nn.Linear(h_feats[0], h_feats[1])
        self.lin3 = nn.Linear(h_feats[1], n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        #x = x.to(torch.float64)
        x = self.lin1(x)
        x = self.relu(x) ## could also be relu ort other
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = torch.sigmoid(x)
        return x


