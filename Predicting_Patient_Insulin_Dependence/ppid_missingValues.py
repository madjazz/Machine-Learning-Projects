#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 16:07:20 2017

filename: ppid_missingValues.py

description: calculate missing values per column

author: Timo Klingler
"""

import ppid_manipulations
import ppid_PCA

import pandas as pd

def missing_values_table(df): 
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum()/len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        return mis_val_table_ren_columns     

missing_vals = missing_values_table(ppid_manipulations.df)
missing_vals.sort_values(by = '% of Total Values', inplace = True)
missing_vals.reset_index(inplace = True)
missing_vals.columns = ["Variable", "Missing Values", "% of Total Values"]

df_mval_pca = pd.merge(missing_vals, ppid_PCA.pca_features, on = ["Variable"])