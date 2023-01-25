# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd

df=pd.read_csv('EEG_Eye_State_Classification.csv') 

'''
Target

1 indicates the eye-closed and
0 the eye-open state.
'''

print('********* show 5 row and columns in data *********')
print(df.head(5))
print('********* show 5 end row and colums in data *********')
print(df.tail())
print('********* show information data ********* ')
print(df.info())
print('********* chek  non value in dataset ********* ')
print(df.isnull().sum())
print('********* The number of repetitions of the value in the data ********* ')
print(df.nunique())
print('*********  Duplicate values in the data ********* ')
print(df.duplicated().sum())


        
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
    
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=(0))
'''
from sklearn.preprocessing import StandardScaler
standardScaler=StandardScaler()
X_train=standardScaler.fit_transform(X_train)
X_test=standardScaler.fit_transform(X_test)
'''
from sklearn.naive_bayes import GaussianNB
class1=GaussianNB()
class1.fit(X_train, y_train)

y_pred=class1.predict(X_test)
    
    
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score

print('naive_bayes :',accuracy_score(y_test, y_pred))


from sklearn.neighbors import KNeighborsClassifier
X_knn_train, X_knn_test, y_knn_train, y_knn_test = train_test_split(X, y, test_size = 0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
standardScaler=StandardScaler()
X_knn_train=standardScaler.fit_transform(X_knn_train)
X_knn_test=standardScaler.fit_transform(X_knn_test)

knn = KNeighborsClassifier(n_neighbors=7)
  
knn.fit(X_knn_train, y_knn_train)

y_knn_pred=knn.predict(X_test)


from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(y_knn_test,y_knn_pred)
from sklearn.metrics import accuracy_score

print('neighbors :',accuracy_score(y_knn_test, y_knn_pred))



















