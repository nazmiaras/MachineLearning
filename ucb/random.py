import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("Ads_CTR_Optimisation.csv")

import random

N = 10000 # Veri sayısı
d = 10 # reklam sayısı

toplam = 0
secilenler = []

for n in range(0,N):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n,ad] #veriler deki n.satır = 1 ise odul 1
    toplam = toplam+odul
print("Toplam Odul :")
print(toplam)
plt.hist(secilenler)
plt.show()