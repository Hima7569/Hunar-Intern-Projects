#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
food_coded=pd.read_csv("food_coded.csv")
food_coded


# In[3]:


food_coded.info()


# In[4]:


food_coded.describe()


# In[5]:


null_counts = food_coded.isnull().sum()
print(null_counts)


# In[6]:


null_counts = food_coded.isnull().sum()
columns_with_nulls = null_counts[null_counts > 0]
for col in columns_with_nulls.index:
    print(f"Column: {col}, Null Values: {columns_with_nulls[col]}, Data Type: {food_coded[col].dtype}")


# In[7]:


numeric_col = food_coded.select_dtypes(include=['number']).columns
numeric_col = list(numeric_col)
x=food_coded[numeric_col].isnull().sum()
print(x)


# In[8]:


categorical_cols = food_coded.select_dtypes('object').columns.tolist()
y=food_coded[categorical_cols].isnull().sum()
print(y)


# In[9]:


duplicate_rows = food_coded.duplicated().any()
duplicate_columns = food_coded.columns.duplicated().any()
if duplicate_rows:
    print("The dataset has duplicate rows.")
else:
    print("The dataset does not have duplicate rows.")
if duplicate_columns:
    print("The dataset has duplicate columns.")
else:
    print("The dataset does not have duplicate columns.")


# In[10]:


from sklearn.impute import SimpleImputer
numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')


# In[12]:


numeric_cols = food_coded.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = food_coded.select_dtypes(include=['object']).columns
food_coded[numeric_cols] = numeric_imputer.fit_transform(food_coded[numeric_cols])
food_coded[categorical_cols] = categorical_imputer.fit_transform(food_coded[categorical_cols])
new_data=food_coded


# In[13]:


new_data


# In[15]:


new_data.isnull().sum()


# In[ ]:




