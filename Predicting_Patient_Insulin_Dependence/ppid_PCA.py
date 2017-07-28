#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 10:25:42 2017

filename: ppid_PCA.py

description: Principal Component Analysis for missing value analysis

author: Timo Klingler
"""

import ppid_manipulations

import math
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

pca = PCA(n_components = 15)
pca.fit(ppid_manipulations.df_dummy)

pca_features = pd.DataFrame(pca.transform(pca.components_, columns = ppid_manipulations.df_dummy.columns)).T
pca_features.reset_index(inplace = True)
pca_features.columns = ["Variable", "Component 1", "Component 2", "Component 3", "Component 4",
                        "Component 5", "Component 6", "Component 7", "Component 8", "Component 9",
                        "Component 10", "Component 11", "Component 12", "Component 13", "Component 14",
                        "Component 15"]

pca_var = pca.explained_variance_ratio_
pca_cumvar = np.cumsum(pca_var)

loadings = pca.transform(ppid_manipulations.df_dummy)