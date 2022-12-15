# -*- coding: utf-8 -*-
"""
Jack Noyes
COMSC230
Dr.Omar Rivera Morales
Program: carbonLinRegJHN.py
Program Details: This program reads a .csv file and makes a linear regression plot 
                 using pandas
"""
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression

co = pd.read_csv("co2_edited_dataset.csv") #creates a pandas using the .csv file 
df = pd.DataFrame(co) #dataframe of the pandas

print(df[['Annual Increase', 'Uncertainty']].describe())


#this code was retrieved from Adarsh Menon on TowardsDataScience.com (https://towardsdatascience.com/linear-regression-in-6-lines-of-python-5e1d0cd05b8d
X = df.iloc[:, 0].values.reshape(-1, 1) #this gets the first column of the dataframe and sets it to the X axis
Y = df.iloc[:, 1].values.reshape(-1, 1) #this gets the second column of the dataframe and sets it to the Y axis
linreg = LinearRegression() #creates a linear regression object
linreg.fit(X,Y) #creates the size of the plot
pred = linreg.predict(X) #this makes a regression line for the plot


plt.scatter(X, Y) #makes scaterplot using the X and Y values 
plt.xlabel("Year")
plt.ylabel("Annual Increase of CO2")
plt.show()


plt.scatter(X, Y) #makes scaterplot using the X and Y values 
plt.plot(X, pred, color='red') #plots the regression line 
plt.xlabel("Year")
plt.ylabel("Annual Increase of CO2")
plt.show()
