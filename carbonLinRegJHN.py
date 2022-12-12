# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
co = pd.read_csv("co2-gr-gl_csv.csv")
df = pd.DataFrame(co)
df[['Year','Annual Increase']]
startYear = 1958
xAxis = [[2]]
print(xAxis.size())
for i in range(58):
    xAxis[0][i] += i
    xAxis[1][i] += startYear
    startYear += 1
   
    
X = xAxis[1]
Y = df.iloc[:, 1].values.reshape(-1, 1)
linear_regressor = LinearRegression()
linear_regressor.fit(X,Y)
Y_pred = linear_regressor.predict(X)


plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()