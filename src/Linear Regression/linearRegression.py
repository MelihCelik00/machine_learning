# -*- coding: utf-8 -*-
"""
@author
This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("dataset/original.csv", sep = ";")

x = df.deneyim.values.reshape(-1, 1)
y = df.maas.values.reshape(-1, 1)

model = LinearRegression()
model.fit(x, y)

y_prediction = model.predict(x)

plt.scatter(x, y)
plt.plot(x, y_prediction, color = "red")
plt.show()
