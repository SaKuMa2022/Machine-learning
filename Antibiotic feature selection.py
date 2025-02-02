#!/usr/bin/env python
# coding: utf-8

# In[8]:



import numpy as np
import pandas as pd


# In[9]:


DF = pd.read_csv(r"C:\Users\.......-out.csv", encoding= 'unicode_escape',index_col=0)


# In[10]:


DF['MIC'] = [4,
16,
16,
32,
32,
16,
16,
16,
34,
32,
4,
8,
34,
34,
8,
8,
8,
34,
16,
16,
8,
16,
8,
8,
8,
34,
4,
34,
4,
4,
8,
8,
4,
]


# In[11]:


DF.head()


# In[12]:


# Feature selection through mutual info regression
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(DF.drop(labels=['MIC'], axis=1),
    DF['MIC'],
    test_size=0.3,
    random_state=0)
from sklearn.feature_selection import mutual_info_regression
mutual_info = mutual_info_regression(X_train, y_train)
mutual_info


# In[13]:


# mutual info values
mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False)


# In[49]:


mutual_info.sort_values(ascending=False)[0:50].plot.bar(figsize=(15,5))


# In[15]:


# selecting the top 20th percentile of features
from sklearn.feature_selection import SelectPercentile
selected_top_columns = SelectPercentile(mutual_info_regression, percentile=20)
selected_top_columns.fit(X_train, y_train)
selected_top_columns.get_support()


# In[16]:


X_train.columns[selected_top_columns.get_support()]


# In[20]:


# Selecting the top 50 features/columns 
new_f= X_train.columns[selected_top_columns.get_support()][0:50]


# In[21]:


new_f


# In[22]:


new_features =['AAAAAAAAAG', 'AAAAAAAACA', 'AAAAAAACGG', 'AAAAAAACTG', 'AAAAAAAGAG',
       'AAAAAAAGTC', 'AAAAAAAGTT', 'AAAAAAATAT', 'AAAAAAATCA', 'AAAAAAATGA',
       'AAAAAAATTC', 'AAAAAACACA', 'AAAAAACACG', 'AAAAAACCGT', 'AAAAAACCTA',
       'AAAAAACCTC', 'AAAAAACGCC', 'AAAAAACGGT', 'AAAAAACTAC', 'AAAAAACTCA',
       'AAAAAACTCC', 'AAAAAACTGG', 'AAAAAAGAAC', 'AAAAAAGAGA', 'AAAAAAGAGG',
       'AAAAAAGATC', 'AAAAAAGATG', 'AAAAAAGCAC', 'AAAAAAGCTG', 'AAAAAAGCTT',
       'AAAAAAGGAG', 'AAAAAAGGCT', 'AAAAAAGGGC', 'AAAAAAGGGT', 'AAAAAAGGTT',
       'AAAAAAGTCC', 'AAAAAAGTCG', 'AAAAAAGTGC', 'AAAAAAGTGG', 'AAAAAAGTTA',
       'AAAAAAGTTG', 'AAAAAATACC', 'AAAAAATACG', 'AAAAAATATA', 'AAAAAATATG',
       'AAAAAATCCT', 'AAAAAATCGC', 'AAAAAATCTT', 'AAAAAATGAA', 'AAAAAATGCT']


# In[25]:


# creating the dataframe with new features
newDF= X_train[new_features]


# In[26]:


newDF.head()


# In[27]:


# using DecisionTreeRegressor model to demonstrate the utility of new features
from sklearn.tree import DecisionTreeRegressor
kmer_model = DecisionTreeRegressor(random_state=1)
kmer_model.fit(newDF,y_train)


# In[32]:


newX_test= X_test[new_features]


# In[33]:


newX_test.head()


# In[36]:


# making predictions with new model using selected set of kmers/features
print('Making predictions for following kmer: ')
print(newX_test.head())
print('The predictions are:')
print(kmer_model.predict(newX_test.head()))


# In[ ]:




