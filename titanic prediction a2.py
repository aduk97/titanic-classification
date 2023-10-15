#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = pd.read_csv('C:\\Users\\adars\\OneDrive\\Desktop\\CERTIFICATIONS\\Titanic-Dataset.csv')
data.head()
data.tail()
data.shape
data.columns
data.isnull().sum()
data.drop(["PassengerId","Name","Ticket","Cabin"],axis=1,inplace=True)
data



# In[ ]:





# In[2]:


from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
imputer = SimpleImputer(strategy = 'most_frequent')
data[['Age','Embarked']]=imputer.fit_transform(data[['Age','Embarked']])
print(data[['Age','Embarked']])


# In[3]:


data.isnull().sum()
label_encoder = LabelEncoder()
data["Sex"] = label_encoder.fit_transform(data["Sex"])
data["Embarked"] = label_encoder.fit_transform(data["Embarked"])
data.describe()
data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),cbar = True, square = True, fmt='.1f',annot=True,cmap='Blues')


# In[4]:


X = data.drop("Survived", axis=1)
y = data["Survived"]
y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("No. of training samples : ",len(X_train))
print("No. of test samples : ",len(X_test))
model = LogisticRegression()
LogisticRegression()
model.fit(X_train, y_train)
y_preds = model.predict(X_test)
y_preds
accuracy = accuracy_score(y_test, y_preds)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_preds))
print("Labels from test set are :",y_test)
print("\nPredicted labels are :",y_preds)
print("Probabilites of prediciton are :",np.round(model.predict_proba(X_test),2))
plt.scatter(y_test,y_preds)
plt.xlabel("Age")
plt.ylabel("Sex")
plt.title("Logistic Regression Decision Boundary for Titanic Classification")
plt.show()


# In[ ]:




