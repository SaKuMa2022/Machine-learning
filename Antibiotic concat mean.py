#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import numpy as np


# In[2]:


DF = pd.read_csv(r"C:\Users\.........-out.csv", encoding= 'unicode_escape',index_col=0)


# In[3]:


# Number of new concatenated columns to create
num_new_columns = 20

# List to hold names of new concatenated columns
new_columns = []

for _ in range(num_new_columns):
    # Select 3 random columns from the 50 columns
    random_columns = np.random.choice(DF.columns, 3, replace=False)
    
    # Create the new column name based on the selected columns
    new_column_name = '_'.join(random_columns)
    
    # Add the new column name to the list
    new_columns.append(new_column_name)
    
    # Sum the selected columns and add it as a new column
    DF[new_column_name] = DF[random_columns].mean(axis=1)


# In[4]:


DF[new_column_name].head()


# In[5]:


DF.head()


# In[6]:


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


# In[7]:


DF.head()


# In[8]:


# Feature selection through mutual info regression
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(DF.drop(labels=['MIC'], axis=1),
    DF['MIC'],
    test_size=0.3,
    random_state=0)
from sklearn.feature_selection import mutual_info_regression
mutual_info = mutual_info_regression(X_train, y_train)
mutual_info


# In[9]:


# mutual info values
mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False)


# In[10]:


mutual_info.sort_values(ascending=False)[0:50].plot.bar(figsize=(15,5))


# In[11]:


# selecting the top 20th percentile of features
from sklearn.feature_selection import SelectPercentile
selected_top_columns = SelectPercentile(mutual_info_regression, percentile=20)
selected_top_columns.fit(X_train, y_train)
selected_top_columns.get_support()


# In[110]:


# Selecting the top features/columns 
new_f= X_train.columns[selected_top_columns.get_support()][0:200]


# In[111]:


new_F2= new_f.to_list()


# In[112]:


# creating the dataframe with new features
newDF= X_train[new_F2]


# In[113]:


newDF.head()


# In[114]:


from numpy import loadtxt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score


# In[115]:


newX_test= X_test[new_F2]


# In[116]:


models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('RFG',RandomForestRegressor(random_state=1)))
models.append(('SVM', SVC()))
models.append(('DTR',DecisionTreeRegressor(random_state=1)))




# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

for name, model in models:
    
    model.fit(newDF, y_train)

    y_pred = model.predict(newX_test.head())
    predictions = [round(value) for value in y_pred]


    # evaluate predictions
    accuracy = accuracy_score(y_test.head(), predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0),name)
    


# In[ ]:




