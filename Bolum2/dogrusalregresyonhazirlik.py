# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 14:56:09 2018

@author: NazmiAras
"""

#1.Kutyphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.Veri Onisleme

#2.1 Veri Yukleme
veriler = pd.read_csv("satislar.csv")

#veri onisleme
aylar = veriler[["Aylar"]]
print(aylar)

satislar = veriler[["Satislar"]]
print(satislar)


#verinin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)
'''
#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
'''

#model inşası (linear regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))

plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")