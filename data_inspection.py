#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:16:34 2024

Data Inspection functions
"""

import data
import seaborn as sns
import matplotlib.pyplot as plt

df_train = data.import_from_csv("train")
df_test = data.import_from_csv("test")
values = ["device_serial", 
          "substance_form_display", 
          "measure_type_display"]


#Purity frequency
plt.figure(figsize=(10, 6)) 
sns.histplot(x='PURITY', kde=True, data=df_train)
plt.title('Frequency of different purities')
plt.xlabel('Purity')
plt.ylabel('Frequency of PURITY')
plt.savefig("Graphs/frequency_of_PURITY")


for value in values:    
    # Purity by value
    plt.figure(figsize=(10, 10)) 
    sns.boxplot(x='PURITY', y=value, data=df_train.sort_values(by="PURITY"))
    plt.title('Distribution of purity time by ' + value)
    plt.xlabel('Purity')
    plt.ylabel(value)
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.9, top=0.9)
    plt.savefig("Graphs/distribution_by_" + value)

    #Frequency of value
    plt.figure(figsize=(10, 6)) 
    sns.histplot(x=value, kde=False, data=df_train)
    plt.title('Frequency of different ' + value)
    plt.xlabel('Purity')
    plt.ylabel('Frequency of ' + value)
    plt.xticks(rotation=65)
    plt.subplots_adjust(left=0.2, bottom=0.3, right=0.9, top=0.9)
    plt.savefig("Graphs/frenquency_of_" + value)


















