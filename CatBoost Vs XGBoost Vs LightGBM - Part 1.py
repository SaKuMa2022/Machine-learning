#!/usr/bin/env python
# coding: utf-8

# In[20]:


# install catboost and xgboost
get_ipython().system('pip install catboost')
get_ipython().system('pip install xgboost')


# In[21]:


# install lightgbm
get_ipython().system('pip install lightgbm')


# In[1]:



import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\Sanjay\Downloads\LE.csv')
df = df [['Country','Year','Status','Life expectancy']]


# In[46]:


df.head()


# In[22]:


from catboost import CatBoostRegressor, Pool
X = df.drop(columns='Life expectancy')
y = df['Life expectancy']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                    random_state=1)

pool_train = Pool(X_train, y_train,
                  cat_features = ['Country','Year','Status'])

pool_test = Pool(X_test, cat_features = ['Country','Year','Status'])


#CatBoost

import time

start = time.time()

cbr = CatBoostRegressor(iterations=100,max_depth=2)

cbr.fit(pool_train)
y_pred = cbr.predict(X_test)

from sklearn.metrics import r2_score as RSquared


cb_rsquared = np.sqrt(RSquared(y_test, y_pred))
print("R Squared for CatBoost: ", np.mean(cb_rsquared))

end = time.time()
diff = end - start
print('Execution time:', diff)


# In[24]:


#XGBoost
import xgboost as xgb
from sklearn import preprocessing
X = df.drop(columns='Life expectancy')
y = df['Life expectancy']

lbl = preprocessing.LabelEncoder()
#Country','Year','Status
X['Country'] = lbl.fit_transform(X['Country'].astype(str))
X['Year'] = lbl.fit_transform(X['Year'].astype(str))
X['Status'] = lbl.fit_transform(X['Status'].astype(str))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                    random_state=1)

start = time.time()
#X_train["Species"].astype("category")
xgbr = xgb.XGBRegressor()

xgbr.fit(X_train, y_train)
y_pred = xgbr.predict(X_test)

xgb_rmse = np.sqrt(RSquared(y_test, y_pred))
print("R Squared for XGBoost: ", np.mean(xgb_rmse))

end = time.time()
diff = end - start
print('Execution time:', diff)


# In[49]:


import lightgbm
X = df.drop(columns='Life expectancy')
y = df['Life expectancy']
obj_feat = list(X.loc[:, X.dtypes == 'object'].columns.values)

for feature in obj_feat:
    X[feature] = pd.Series(X[feature], dtype="category")

start = time.time()

lgbmr = lightgbm.LGBMRegressor()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                    random_state=1)

lgbmr.fit(X_train, y_train)
y_pred = lgbmr.predict(X_test)

lgbm_rsquared = np.sqrt(RSquared(y_test, y_pred))
print()
print("R squared for LightGBM: ", np.mean(lgbm_rsquared))

end = time.time()
diff = end - start
print('Execution time:', diff)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 1. Community support/Performance/Want to tune - XGBOOST
# # 2. Hurry to train/Performance/Dont want to tune - LightGBM
# # 3. More categorical, GPU, Large data - CatBoost

# In[ ]:




