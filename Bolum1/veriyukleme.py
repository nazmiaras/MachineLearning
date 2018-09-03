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
veri = pd.read_csv("veriler.csv")
boy = veri[["boy"]]
print(boy)
boykilo= veri[["boy","kilo"]]
print(boykilo)