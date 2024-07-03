#!/usr/bin/env python
# coding: utf-8

# In[122]:


import numpy as np
import pandas as pd


# In[123]:


house=pd.read_csv("house price data.csv")
house


# In[124]:


house.info()


# In[125]:


house.describe()


# In[126]:


house.isnull().sum()


# In[127]:


duplicate_rows = house.duplicated().any()
duplicate_columns = house.columns.duplicated().any()
if duplicate_rows:
    print("The dataset has duplicate rows.")
else:
    print("The dataset does not have duplicate rows.")
if duplicate_columns:
    print("The dataset has duplicate columns.")
else:
    print("The dataset does not have duplicate columns.")


# In[128]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(house, test_size=0.2, random_state=42)
train


# In[129]:


test


# In[130]:


print('train_df.shape :', train.shape)
print('val_df.shape :', test.shape)


# In[131]:


input_cols = list(train.columns)[3:-1]
target_col = 'price'


# In[132]:


train_inputs = train[input_cols].copy()
train_targets = train[target_col].copy()
test_inputs = test[input_cols].copy()
test_targets = test[target_col].copy()


# In[ ]:





# In[133]:


numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()


# In[134]:


train_inputs[numeric_cols].describe()


# In[135]:


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoder.fit(house[categorical_cols])


# In[136]:


encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
print(encoded_cols)


# In[137]:


train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])


# In[138]:


pd.set_option('display.max_columns', None)
test_inputs


# In[139]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_inputs[numeric_cols + encoded_cols], train_targets)


# In[140]:


predictions=model.predict(train_inputs[numeric_cols + encoded_cols])
predictions


# In[141]:


def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))
loss = rmse(train_targets, predictions)
print('Loss:', loss)


# In[142]:


pred=model.predict(test_inputs[numeric_cols+encoded_cols])
pred


# In[143]:


losss=rmse(test_targets,pred)
losss


# In[144]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# In[145]:


mae = mean_absolute_error(test_targets, pred)
mae

