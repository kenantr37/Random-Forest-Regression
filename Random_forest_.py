# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:11:35 2020

@author: Zeno
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
""" 
In our sample, there is a theatre and y is price of the seat
x is location of the seat.When seat's location gets closer to the scene,
price of the seat increases.We want to predict seat price.
"""
x=np.array([10,9,8,7,6,5,4,3,2,1]).reshape(-1,1) #Location of seats
y=np.array([5,10,16,23,32,41,50,61,75,100]) #Prices of seats
#Let's create Random forest model and fit x,y
random_forest_model = RandomForestRegressor(n_estimators=100,random_state=(42)).fit(x,y)
#as you can see, estimators numbers are 100
#and we don't want to get different predictions every time and for this our random state is 42

#Let's look at the prediction for seat's location is 7
print(random_forest_model.predict([[7]])) #prediction=24.6 <-> real price=23

x_gradually = np.arange(min(x),max(x),0.01).reshape(-1,1) #we want to predict and visualize gradually
y_head = random_forest_model.predict(x_gradually)#prediction variable
#visualization
plt.scatter(x,y,color = "black")
plt.plot(x_gradually,y_head,color ="purple",label ="Random Forest Regression")
plt.xlabel("location of the seat")
plt.ylabel("price of the seat")
plt.grid(True)
plt.legend()
plt.show()