#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:04:31 2024

Fonctions for .cvs files and panda objects
"""

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)



def import_from_csv(csv_file_name, import_from_Results=False):
    if import_from_Results :
        folder = "Results"
    else : 
        folder = "Data"
    return pd.read_csv(folder + "/" + csv_file_name + ".csv")


def export_as_csv(pd_object):
    pd_object.to_csv('tmp/dataframe.csv', index=False)
    
    
def labels(pd_object):
    return [col for col in pd_object]


def unique_in_col(pd_object_col):
    liste = []
    for i in pd_object_col:
        if i not in liste:
            liste.append(i)
    return liste


df = import_from_csv("train")
df2 = import_from_csv("train")
labels = labels(df)

sample_name_index = unique_in_col(df.sample_name)
device_serial_index  = unique_in_col(df.device_serial)
substance_form_display_index  = unique_in_col(df.substance_form_display)
measure_type_display_index  = unique_in_col(df.measure_type_display)
prod_substance_index  = unique_in_col(df.prod_substance)

























