# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 11:42:41 2018

@author: NazmiAras
"""
#1.Kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme
#2.1. Veri Yukleme

veriler = pd.read_csv("veriler.csv")
#veriler = pd.read_csv("veriler.csv")

x = veriler.iloc[:,1:4]
y = veriler.iloc[:,4:]

#Verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)

#Verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)
Y_pred = logr.predict(X_test);
print(Y_pred)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,Y_pred);
print(cm)

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=1,metric='minkowski')
kn.fit(X_train,y_train)
Y_pred = kn.predict(X_test)

print("KNN :")
cm = confusion_matrix(y_test,Y_pred);
print(cm)

from sklearn.svm import SVC

svc = SVC(kernel='linear')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)
print("SVC")
cm = confusion_matrix(y_test,y_pred)

print(cm)
