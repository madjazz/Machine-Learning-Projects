#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:17:48 2017

filename: ppid_manipulations

description: 

author: Timo Klingler
"""

import ppid_credentials

import pandas as pd

demographic = pd.read_csv(ppid_credentials.data_path + 'demographic.csv')
diet = pd.read_csv(ppid_credentials.data_path + 'diet.csv')
examination = pd.read_csv(ppid_credentials.data_path + 'examination.csv')
labs = pd.read_csv(ppid_credentials.data_path + 'labs.csv')
questionnaire = pd.read_csv(ppid_credentials.data_path + 'questionnaire.csv')

demographic = demographic[['SEQN', 'RIDAGEYR', 'RIAGENDR']]
examination = examination[['SEQN', 'BMXBMI']]
diet = diet[['SEQN', 'DR1TKCAL', 'DR1TPROT', 'DR1TCARB', 'DR1TSUGR', 'DR1TALCO', 'DR1TTFAT']]
questionnaire = questionnaire[['SEQN', 'DIQ050']]

step1 = pd.merge(demographic, examination, on = ['SEQN'])
step2 = pd.merge(step1, diet, on = ['SEQN'])
step3 = pd.merge(step2, questionnaire, on = ['SEQN'])
df = pd.merge(step3, labs, on = ['SEQN'])

"""
Create Dummy Variables
"""

df_dummy = df
df_dummy[df_dummy.notnull()] = 0
df_dummy = df.fillna(1)

"""
Join missing values counts with pca
"""

