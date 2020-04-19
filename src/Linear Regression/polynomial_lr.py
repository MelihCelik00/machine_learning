# -*- coding: utf-8 -*-
# @author: Melih Safa Celik

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Dataset/poly_linearr.csv',sep=";")

y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("Max Speed")
plt.xlabel("Price")
#plt.show()

"""
linear regression: y = b0 + b1*x
multiple linear regression: y = b0 + b1*x1 + b2*x2
polynomial linear regression: y= b0 + b1*x + b2*x^2 + ... + bn*x^n
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
polynomial_regression = PolynomialFeatures(degree=2)

x_polynomial = polynomial_regression.fit_transform(x)

linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)

y_head2 = linear_regression2.predict(x_polynomial)

plt.plot(x,y_head2,color="blue",label="poly")

#from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x,y)

y_head = lr.predict(x)
plt.plot(x,y_head, color = "red",label="linear")
plt.legend()

plt.show()

#print(lr.predict(10000))
