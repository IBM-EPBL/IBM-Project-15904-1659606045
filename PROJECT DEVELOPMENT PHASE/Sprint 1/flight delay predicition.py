#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error


# In[2]:


dataset = pd.read_csv('flightdata.csv')
dataset.head()


# In[3]:


dataset.columns


# In[4]:


data = dataset.drop(['YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK',
       'UNIQUE_CARRIER', 'TAIL_NUM', 'FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN',
       'DEST_AIRPORT_ID', 'DEST', 'DISTANCE', 'Unnamed: 25'], axis=1)
data.head()


# In[5]:


data.dtypes


# In[6]:


data.shape


# In[7]:


data.isna().sum()


# In[8]:


data.dropna(inplace=True)
data.isna().sum()


# In[9]:


data.shape


# In[10]:


# convert to categorical
data['CANCELLED'] = data['CANCELLED'].astype(object)
data['DIVERTED'] = data['DIVERTED'].astype(object)
data['ARR_DEL15'] = data['ARR_DEL15'].astype(object)
data['DEP_DEL15'] = data['DEP_DEL15'].astype(object)
data.dtypes


# In[13]:


X = data[['CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY',
       'DEP_DEL15', 'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY',
       'CANCELLED', 'DIVERTED', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME']]
Y = data[['ARR_DEL15']]


# In[14]:


# splitting into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[15]:


# create a regressor object
regressor = DecisionTreeRegressor(random_state = 0) 


# In[16]:


# fit the regressor with X and Y data
regressor.fit(X_train, y_train)


# In[17]:


# prediction
y_pred = regressor.predict(X_test)


# In[18]:


print("MSE for DTReg (All features): " , np.round(mean_squared_error(y_test, y_pred), 2))


# In[19]:


joblib.dump(regressor, 'model1.pkl')


# In[ ]:




