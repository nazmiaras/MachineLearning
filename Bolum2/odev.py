# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 14:56:09 2018

@author: NazmiAras
"""

#1.Kutyphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#2.Veri Onisleme

#2.1 Veri Yukleme
veriler = pd.read_csv("odev_tenis.csv")

#veri onisleme


#endoer: Kategorik->Numeric

veriler2 = veriler.apply(LabelEncoder().fit_transform)

c=veriler2.iloc[:,:1]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
c=ohe.fit_transform(c).toarray()
print(c)

havadurumu =pd.DataFrame(data=c,index=range(14),columns=["o","r","s"])

sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis = 1)

sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler],axis = 1)

#verinin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33,random_state=0)


#model inşası (linear regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

boy = s2.iloc[:,3:4].values
print(boy)
sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)

x_train,x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33,random_state=0)

r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)

import statsmodels.formula.api as sm

X = np.append(arr=np.ones((22,1)).astype(int),values = veri,axis=1)
X_l = veri.iloc[:,[0,1,2,3,4,5]].values
r_ols=sm.OLS(endog = boy,exog=X_l)
r = r_ols.fit()
print(r.summary())

X = np.append(arr=np.ones((22,1)).astype(int),values = veri,axis=1)
X_l = veri.iloc[:,[0,1,2,3,5]].values
r_ols=sm.OLS(endog = boy,exog=X_l)
r = r_ols.fit()
print(r.summary())
