#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:38:50 2024

@author: maximelehmann
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel





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




"""
unwanted = ["prod_substance", "measure_type_display", "substance_form_display", "sample_name", "device_serial"]
df_preprocess_train, df_preprocess_test = preprocessing(df_train, df_test, unwanted, tresh) 
   

X = df_preprocess_train.iloc[:, 1:]
y = df_preprocess_train.PURITY
"""

X = df_train.iloc[:, 6:]
y = df_train.PURITY

# Standardize the input features
standardizer = StandardScaler()
X_standardized = standardizer.fit_transform(X)
X_tensor = torch.tensor(X_standardized[:1000, :], dtype=torch.float32)
y_tensor = torch.tensor(y[:1000], dtype=torch.float32).reshape(-1, 1)

# Define the neural network model using PyTorch
class NN_model(nn.Module):
    def __init__(self, input_size=X.columns.size, n_neurons=32, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, n_neurons),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(n_neurons, 1)
        )

    def forward(self, x):
        return self.layers(x)
    
# create model with skorch
model_skorch = NeuralNetRegressor(
    NN_model,
    criterion=nn.MSELoss,
    optimizer=optim.Adam,
    max_epochs=90,
    batch_size=32,
    verbose=False
)


# Define the parameter grid for hyperparameter tuning
param_grid = {
    'module__n_neurons': [32, 64, 128],
    'module__dropout_rate': [0, 0.1, 0.2]
}


# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=model_skorch, param_grid=param_grid, cv=5)


grid_result = grid_search.fit(X_tensor, y_tensor)

print("Best MSE: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Get the best model from the grid search
mach2 = grid_result.best_estimator_


# Fit the best model to the data
mach2.fit(X_tensor, y_tensor)


# Make predictions
predictions = mach2.predict(torch.tensor(X_standardized[1000:, :], dtype=torch.float32))


mach2.get_params()


train_rmse = np.sqrt(mean_squared_error(y.iloc[1000:],predictions))
                                         



print("tresh = " + str(tresh))
print("RMSE = " + str(train_rmse))
print(" ")


X_test_standardized = standardizer.fit_transform(df_test.iloc[:, 5:])

# Create dataframe with indices and purity
indices = df_test.index + 1
purity =  mach2.predict(torch.tensor(X_test_standardized, dtype=torch.float32))

df = pd.DataFrame({
    'ID': indices,
    'PURITY': purity.flatten()
})

df.to_csv('submission_neural_net.csv', index=False)



