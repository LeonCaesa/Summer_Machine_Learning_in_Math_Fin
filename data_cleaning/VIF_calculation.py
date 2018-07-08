#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 10:10:02 2018

@author: thelightofking
"""

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
data=pd.read_sas("variables.sas7bdat")
time=pd.to_timedelta(data['DATE'], unit='D') + pd.datetime(1960, 1, 1)
data['Date']=time.values
data=data[data['fyear']>2015]

data=data.fillna(data.mean(),inplace = True)
data=pd.DataFrame(data)


non_float=data.dtypes[data.dtypes!='float64'].index
data_float=data.drop(non_float,axis=1)
data_float=data_float.dropna(how='any',axis=1)


data_float['Date']=data['Date']
a=data_float.groupby(['permno','Date'])
return_=a.agg(np.mean)


vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(data_float.values, i) for i in range(data_float.shape[1])]
vif["features"] = data_float.columns
