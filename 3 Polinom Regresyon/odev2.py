# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 05:41:07 2018

@author: NazmiAras
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm


# veri yukleme
veriler = pd.read_csv('maaslar_yeni.csv')

#data frame dilimleme (slice)
x = veriler.iloc[:,2:5]
y = veriler.iloc[:,5:]

#NumPY dizi (array) dönüşümü
X = x.values
Y = y.values


#linear regression
#doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

model1 = sm.OLS(lin_reg.predict(X),X)
print(model1.fit().summary())

print("Linear Regression R2 Value:")
print(r2_score(Y,lin_reg.predict(X)))


#polynomial regression
#doğrusal olmayan (nonlinear model) oluşturma
from sklearn.preprocessing import PolynomialFeatures
# 4. dereceden polinom
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)

print("Poly OLS")
model2 = sm.OLS(lin_reg3.predict(poly_reg3.fit_transform(X)),X)
print(model2.fit().summary())

print("Polynomial Regression 4 R2 Value:")
print(r2_score(Y,lin_reg3.predict(x_poly3)))



#verilerin olceklenmesi

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
y_olcekli = sc1.fit_transform(Y)

from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)


print("SVR OLS")
model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())

print("SVR R2 Value:")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))


from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)


print("DT OLS")
model4 = sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())


print("Decision Tree R2 Value:")
print(r2_score(Y,r_dt.predict(X)))


from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y)


print("RF OLS")
model5 = sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())


print("Random Forest R2 Value:")
print(r2_score(Y,rf_reg.predict(X)))

print("------------------")

print("Linear Regression R2 Value:")
print(r2_score(Y,lin_reg.predict(X)))


print("Polynomial Regression 4 R2 Value:")
print(r2_score(Y,lin_reg3.predict(x_poly3)))

print("SVR R2 Value:")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))


print("Decision Tree R2 Value:")
print(r2_score(Y,r_dt.predict(X)))


print("Random Forest R2 Value:")
print(r2_score(Y,rf_reg.predict(X)))