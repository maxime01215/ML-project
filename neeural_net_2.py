#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 19:12:13 2024

@author: maximelehmann
"""


from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import warnings
from sklearn.model_selection import KFold
import optuna
import torch.utils
import matplotlib.pyplot as plt
import skorch

df_train = pd.read_csv('Data/train.csv')
df_test = pd.read_csv('Data/test.csv')
seed = 1

def rand_forest(df_train, thresh):
    
    X = df_train.iloc[:, 6:]
    y = df_train[['PURITY']].squeeze()

    # Create and train random forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=seed) 
    rf_model.fit(X, y)

    # We select features depending on their importances
    sfm = SelectFromModel(rf_model, threshold=thresh)

    # we get the index of the unwanted features
    feature_indices = ~sfm.get_support()

    # we obtain the names of the unwanted features
    removed_columns_names = X.columns[feature_indices]
   
    return removed_columns_names


def preprocessing(df_train, df_test, unwanted, thresh):

    #we use random forest to remove unwanted spectrum
    removed_columns_names = rand_forest(df_train, thresh)
    
    df_train = df_train.drop(columns=removed_columns_names)
    df_test = df_test.drop(columns=removed_columns_names)
    
    #remove unwwanted columns
    df_train = df_train.drop(unwanted, axis=1)
    df_test = df_test.drop(unwanted, axis=1)

    return df_train, df_test 

tresh = 0.00206913808111479


unwanted = ["prod_substance", "measure_type_display", "substance_form_display", "sample_name", "device_serial"]
df_preprocess_train, df_preprocess_test = preprocessing(df_train, df_test, unwanted, tresh) 
   

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row_X = self.X.iloc[idx]
        row_y = self.y.iloc[idx]
        x = torch.tensor(row_X, dtype=torch.float)
        y = torch.tensor(row_y, dtype=torch.float)
        return x, y

# hyperparameters
num_epochs = 90
batch_size = 32

# Network

class Network(nn.Module):
    def __init__(self, N_init, N1, N2, N3, N4):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(N_init, N1)
        self.linear2 = nn.Linear(N1, N2)
        self.linear3 = nn.Linear(N2, N3)
        self.linear4 = nn.Linear(N3, N4)
        self.linear5 = nn.Linear(N4, 1)  # output : 1 for Retention time

    def forward(self,x):
        z = F.selu(self.linear1(x))
        z = F.selu(self.linear2(z))
        z = F.selu(self.linear3(z))
        z = F.selu(self.linear4(z))
        z = self.linear5(z)
        return z

class EarlyStopTraining:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, loss, model):
        if self.best_loss is None:
            self.best_loss = loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0

## loss
def loss(y_pred, y_hat):
    return torch.sqrt(torch.mean((y_pred - y_hat) ** 2))

# Ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Objective function for Bayesian optimization
def objective(trial):
    # Hyperparameter Suggestion
    N1 = trial.suggest_int('N1', 80, 170)
    N2 = trial.suggest_int('N2', 50, 120)
    N3 = trial.suggest_int('N3', 20, 80)
    N4 = trial.suggest_int('N4', 5, 20)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    lab_factor = trial.suggest_int('lab_factor', 5, 15)
    
    
    # Parameters for cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train_index, val_index in kf.split(df_preprocess_train):
        # Split data based on indices
        df_fold_train = df_preprocess_train.iloc[train_index]
        df_process_val =  df_preprocess_train.iloc[val_index]


    y_train = df_fold_train['RT']
    X_train = df_fold_train.drop(['RT'], axis=1)
    X_train.iloc[:, -24:] = X_train.iloc[:, -24:] * lab_factor

    dataset = CustomDataset(X_train, y_train)
    # Creating the DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Modifying the model to use suggested hyperparameters
    model = Network(X_train.shape[1] ,N1, N2, N3, N4)
    

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # We stop gradient descent on the training loss just to avoid unnecessary computation
    # Note that this is different from standard early stopping on a validation set as seen in class!
    early_stopping = EarlyStopTraining(patience=10)

    # training loop
    for epoch in range(num_epochs):
        train_loss_avg = 0
        num_batches = 0

        for batch in data_loader:
            x, y_true = batch
            x, y_true = x, y_true

            # Forward propagation 
            y_pred = model(x)

            # Loss computation
            loss_value = loss(y_pred, y_true.unsqueeze(-1))

            # Backpropagation and update weight
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            # Sum average loss
            train_loss_avg += loss_value.item()
            num_batches += 1

        train_loss_avg /= num_batches

        early_stopping(loss_value, model)

    if early_stopping.early_stop: model.load_state_dict(early_stopping.best_model)
    
    model.eval()
    y_val = df_process_val['RT']
    X_val = df_process_val.drop(['RT'], axis=1)
    X_val.iloc[:, -24:] = X_val.iloc[:, -24:] * lab_factor

    x = torch.zeros(X_val.shape[0] * X_val.shape[1]).reshape(X_val.shape[0], X_val.shape[1])
    for idx in range(0,X_val.shape[0]):
        row = X_val.iloc[idx]
        x[idx] = ( torch.tensor((row), dtype=torch.float) )

    y_pred = model(x)
    # RMSE
    val_loss = loss(y_pred, torch.tensor(y_val.values).unsqueeze(-1))
    return val_loss


# Creating a study object and running the optimization
study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=seed))
study.optimize(objective, n_trials=20)

# Displaying the best hyperparameters
print("Best hyperparameters: ", study.best_params)

N1 = study.best_params['N1']
N2 = study.best_params['N2']
N3 = study.best_params['N3']
N4 = study.best_params['N4']
learning_rate = study.best_params['learning_rate']
lab_factor = study.best_params['lab_factor']

y_train = df_preprocess_train['RT']
y_train = torch.from_numpy(y_train.values.astype(np.float32)).unsqueeze(-1)
X_train = df_preprocess_train.drop(['RT'], axis=1)
X_train.iloc[:, -24:] = X_train.iloc[:, -24:] * lab_factor
X_train = torch.from_numpy(X_train.values.astype(np.float32))

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sknn = skorch.NeuralNet(module = Network,
                        module__N_init = X_train.shape[1],
                        module__N1 = study.best_params['N1'],
                        module__N2 = study.best_params['N2'],
                        module__N3 = study.best_params['N3'],
                        module__N4 = study.best_params['N4'],
                        criterion = torch.nn.MSELoss,
                        optimizer = torch.optim.Adam,
                        optimizer__lr = learning_rate,
                        optimizer__weight_decay = 1e-5,
                        batch_size = batch_size,
                        train_split = None,
                        max_epochs = num_epochs)

sknn.fit(X_train, y_train)


plt.ion()

fig = plt.figure()
plt.plot(sknn.history[:, 'train_loss'], label = "Training Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
































