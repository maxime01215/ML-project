#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 17:28:41 2024

@author: maximelehmann
"""

import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


df_train = pd.read_csv('Data/train.csv')
df_test = pd.read_csv('Data/test.csv')
seed = 1

spectrum_train = df_train.iloc[:, 6:]
spectrum_filtered_train = pd.DataFrame(savgol_filter(spectrum_train, 7, 3, deriv = 2, axis = 1))
spectrum_filtered_standardized_train = zscore(spectrum_filtered_train, axis = 1)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(spectrum_filtered_standardized_train, df_train.PURITY, test_size=0.3, random_state=seed)

# Train the Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=seed)
rf.fit(X_train, y_train)

# Evaluate the model
score_before = rf.score(X_test, y_test)
print(f'Accuracy before feature selection: {score_before:.2f}')

# Extract feature importances
importances = rf.feature_importances_
feature_names = spectrum_filtered_standardized_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Rank features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# Select top N features
top_features = feature_importance_df['Feature'][:10].values
X_train_selected = X_train[top_features].sort_index(axis=1)
X_test_selected = X_test[top_features].sort_index(axis=1)


"""
accuracies = []
for N in range(5, 51, 5):
    
    # Select top N features
    top_features = feature_importance_df['Feature'][:N].values
    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]
    
    # Train the Random Forest model with selected features
    rf_selected = RandomForestRegressor(n_estimators=100, random_state=seed)
    rf_selected.fit(X_train_selected, y_train)
    
    # Evaluate the model
    accuracy_after = rf_selected.score(X_test_selected, y_test)
    accuracies.append(accuracy_after)
    print(f'Accuracy selecting {N:.2f} features : {accuracy_after:.2f}')
"""










