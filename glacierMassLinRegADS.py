#Name: Alden Sahi
#Professor: Dr. Morales
#Program: glacierMassLinRegADS.py
#Program Details: makes a linear regression graph displaying the trend of glacier mass

import numpy as np
import pandas as pd 
import matplotlib.pyplot as pyplot 
from sklearn.linear_model import LinearRegression

#imports data set
csv = pd.read_csv("meanCumulativeGlacierMass.csv") 
#puts data set into df
df = pd.DataFrame(csv) 

#this code was retrieved from Adarsh Menon on TowardsDataScience.com (https://towardsdatascience.com/linear-regression-in-6-lines-of-python-5e1d0cd05b8d)
x_axis = df.iloc[:, 0].values.reshape(-1, 1) #this gets the first column of the dataframe and sets it to the X axis
y_axis = df.iloc[:, 1].values.reshape(-1, 1) #this gets the second column of the dataframe and sets it to the Y axis
#creates linear regression object so we can add attributes
slope = LinearRegression()
#machine learns so we can use predict function later
slope.fit(x_axis, y_axis) 
#this makes a line of regression for the plot based on what it has observed from data set
lineofreg = slope.predict(x_axis) 

#adds scatter graph to pyplot
pyplot.scatter(x_axis, y_axis)
#adds linear regression trend to pyplot
pyplot.plot(x_axis, lineofreg, color='purple') 
#displays graphic
pyplot.show()

