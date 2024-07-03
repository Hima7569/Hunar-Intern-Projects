#!/usr/bin/env python
# coding: utf-8

# In[193]:


import pandas as pd
import numpy as np


# In[194]:


df=pd.read_csv("breast cancer.csv")
df


# In[195]:


df.isnull().sum()


# In[196]:


X=df.drop(columns=['diagnosis','id'])
X


# In[197]:


Y=df['diagnosis']
Y


# In[198]:


from sklearn.model_selection import train_test_split,cross_val_score 
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=42,test_size=0.2)


# In[199]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[200]:


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report 


# In[201]:


from sklearn.neighbors import KNeighborsClassifier


# In[202]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

k_values = range(1, 21)
cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())



# In[203]:


# Plot the cross-validation scores
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Cross Validation Scores')
plt.show()


# In[204]:


best_k = k_values[np.argmax(cv_scores)]
best_k


# In[206]:


knn = KNeighborsClassifier(n_neighbors=best_k)

# Train the model
knn.fit(x_train, y_train)


# In[213]:


train_accuracy=knn.score(x_train,y_train)
train_accuracy


# In[214]:


test_accuracy=knn.score(x_test,y_test)
test_accuracy


# In[215]:


Y_pred = knn.predict(x_test)
test_accuracy = accuracy_score(y_test, Y_pred)
precision = precision_score(y_test, Y_pred, pos_label='M') 
recall = recall_score(y_test, Y_pred, pos_label= 'M')
f1 = f1_score(y_test, Y_pred, pos_label='M')


# In[223]:


print(recall)
print(f1)


# In[222]:


print(precision)


# In[224]:


a=4
print(f'Best k value: {a}')


# In[225]:


#classification_report


# In[230]:


report=classification_report(y_test,Y_pred,target_names=['B','M'])
print(report)


# In[231]:


print(y_test,Y_pred)

