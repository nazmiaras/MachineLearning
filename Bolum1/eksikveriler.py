# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 11:42:41 2018

@author: NazmiAras
"""
#Kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Kodlar
#Veri Yukleme

veriler = pd.read_csv("eksikveriler.csv")
#veriler = pd.read_csv("veriler.csv")
boy = veriler[["boy"]]
print(boy)
boykilo= veriler[["boy","kilo"]]
print(boykilo)

x = 10

class insan:
    boy = 180
    def kosmak(self,b):
        return b+10
    
ali = insan()
print(ali.boy)
print(ali.kosmak(90))

#eksik veriler
#sci-kit learn

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)