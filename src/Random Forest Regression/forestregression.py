# -*- coding: utf-8 -*-
# @author: Melih Safa Celik

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#dataset can be found in Decision Tree folder

df = pd.read_csv("../Decision Tree/Dataset/original.csv",sep=";",header=None)
x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)
#print(x)
#print(y)

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(x,y)
"""
print("7.8 seviyesinde fiyat: ", rfr.predict([[7.8]]))
print("3.4 seviyesinde fiyat: ", rfr.predict([[3.4]]))
print("6 seviyesinde fiyat: ",   rfr.predict([[6]]))
"""
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1) # Don't forget to initialise step size
y_head = rfr.predict(x_)

plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("Tribune Level")
plt.ylabel("Cost")
plt.show()
