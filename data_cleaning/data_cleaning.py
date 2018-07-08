
# coding: utf-8

# In[47]:


# import necessary packages

import pandas as pd
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

import seaborn as sns
sns.set(color_codes=True)


# # 1. Data Cleaning
# 
# - selected the column that we are interested in
# - delete all records with missing indicator ( hard to fill in)
# - fill in missing data with median of the corresponding column
# 
# 

# In[8]:


# Read in data
data = pd.read_sas('rpsdata_rfs.sas7bdat')


# In[9]:


# 1. keep column needed
col_selected = pd.read_excel("selected_column.xlsx")
col_set = col_selected['col_name']
data_selected = data.loc[:,col_set]


# In[10]:


# 2. delete missing indicator
indx_null_indicator = np.isnan(data_selected["rd"])
data_delete_nanindicator = data_selected.loc[~indx_null_indicator, :].reset_index(drop=True)


# In[11]:


# 3. Use mean to fill missing value
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(data_delete_nanindicator)
clean_data = imp.transform(data_delete_nanindicator)
clean_data_df = pd.DataFrame(clean_data,columns = data_delete_nanindicator.columns)


# In[12]:


clean_data_df.head()


# In[13]:


dataset = clean_data_df.copy()

# correct the date column
temp = pd.to_timedelta(dataset['DATE'], unit='D') + pd.datetime(1960, 1, 1)
dataset['DATE']=temp.values

# add year and month column
dataset['year'] = pd.DatetimeIndex(dataset['DATE']).year
dataset['month'] = pd.DatetimeIndex(dataset['DATE']).month


# In[15]:


# delete data before 1988 - keep 30year's data
clean_data_after1988 = dataset[dataset['year']>=1988].reset_index(drop=True)


# delete data with market cap < 500million 
# for a company, all the records after its market cap hit 500m for the frist time will be kept
def dataprocess_mktcap(groupdata):
    groupdata = groupdata.sort_values(['DATE'])
    indx = next((i for i in range(len(groupdata['mve_m'])) if groupdata['mve_m'].values[i] >=500000),len(groupdata['mve_m']))
    groupdata = groupdata.iloc[indx:,:] 
    return groupdata

    
clean_data_mktcap = clean_data_after1988.groupby(['permno'], as_index=False).apply(dataprocess_mktcap)
clean_data_mktcap = clean_data_mktcap.reset_index(drop = True)


# # 2.1 Factor Prepare
# 
# - Assign the factors into different categories (not very accurate here)

# In[57]:


col_info= pd.read_excel("selected_column.xlsx")
col_info.head()

# create dictionary for acronym
factor_dicts = {}
keys = col_info['col_name'].values.tolist()
values = col_info['description'].values.tolist()
for i in range(len(keys)):
    factor_dicts[keys[i]] = values[i]


col_info.set_index(['factor_type','col_name'])



# In[86]:


factor_type = col_info[['factor_type','col_name']].groupby(['factor_type']).count()
factor_type


# # 2.2 Check Data Distribution by factor type

# In[89]:


def distribution_check(factor_type_name):
    check_col = col_info[col_info['factor_type']== factor_type_name]['col_name']
    for col in check_col:
        plt.figure()
        sns.distplot(clean_data_mktcap[col])
        plt.title(factor_dicts[col])

        plt.show()
    


# In[90]:


# type - value
distribution_check('value')


# In[91]:


# type - liquidity
distribution_check('liquidity')
    
# for col in check_col:
#     fig = plt.figure()
#     ax1 = fig.add_subplot(121)
#     ax2 = fig.add_subplot(122)

#     sns.distplot(clean_data_mktcap[col], ax=ax1)
#     sns.regplot(x= col , y='RET', data=clean_data_mktcap, ax=ax2)
#     plt.suptitle(factor_dicts[col])
#     plt.show()


# In[92]:


# type - change
distribution_check('change')


# In[93]:


distribution_check('dividend')


# In[94]:


distribution_check('momentum')


# In[95]:


distribution_check('size')


# In[96]:


distribution_check('volatility')


# In[97]:



distribution_check('quality-accrual')


# In[98]:


distribution_check('quality-balancesheet')


# In[99]:


distribution_check('quality-forecast')                 


# In[100]:


distribution_check('quality-investment')


# In[101]:


distribution_check('quality-profit')


# In[102]:


distribution_check('quality-ratio')


# In[103]:


distribution_check('quality-other')

