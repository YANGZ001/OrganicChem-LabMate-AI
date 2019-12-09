# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:49:48 2019

@author: Lenovo
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.externals.joblib import dump
import matplotlib.pyplot as plt
import time
from sklearn import model_selection
from sklearn.calibration import calibration_curve 
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

dfs = pd.read_excel('/Users/lenovo/desktop/project/discrete variable/Raw data compiled HTE.xlsx', sheet_name=None)
df = dfs['Normalized yields']




ligandes =  np.arange(1,21).tolist()*6*4
bases =   ([1]*20 + [2]*20 + [3]*20 + [4]*20 + [5]*20 + [6]*20)*4 
substrate = [1]*120 + [2]*120 + [3]*120 + [4]*120 

# Define a dictionary containing Students data 
data = {'ligandes': ligandes, 
        'bases': bases ,
        'substrate': substrate ,
        'yield': df['yield'].values } 
# Convert the dictionary into DataFrame 
df_new = pd.DataFrame(data) 


array = df_new.values
X = array[:,:-1] 
Y = array[:,-1] 



nbr_samples = 100
random_data = df_new.sample(n=nbr_samples, random_state=1) # do i need to sample with specific way : like samplefrom ligande ? 
df_random_data = pd.DataFrame(random_data)


array_random = df_random_data.values
X_sample = array_random[:,:-1] 
Y_sample = array_random[:,-1] 



df_train_corrected = df_random_data.iloc[:,:-1]
unseen = pd.concat([df_new.iloc[:,:-1], df_train_corrected]).drop_duplicates(keep=False)
X2 = unseen.values

df_all_combos2 =  df_new.iloc[:,:-1]
df_all_combos2.head()

df_new.head()