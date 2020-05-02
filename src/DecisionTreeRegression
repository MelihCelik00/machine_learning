# -*- coding: utf-8 -*-
# @author: Melih Safa Celik

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Dataset/original.csv", sep=";", header=None)

x = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values.reshape(-1, 1)

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(x, y)

#tree_reg.predict(6)

y_head = tree_reg.predict(x)

plt.scatter(x,y,color="red")
plt.plot(x,y_head,color="green")
plt.xlabel("Tribune_level")
plt.ylabel("Cost")
plt.show()

