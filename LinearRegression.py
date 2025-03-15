import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math


# Read the house price csv file
data = pd.read_csv("Datasets\Datasets\house_prices.csv")
size = data['sqft_living15'] #feature
price = data['price']      # take usefull columns

# machine learning handles arrays not data frames , hence convert it to array
x= np.array(size).reshape(-1,1)      #.reshape(rows,columns)
y= np.array(price).reshape(-1,1)     #Pass -1 as the value, and NumPy will calculate this number for you.



#Make model
model = LinearRegression()
model.fit(x,y)    #used gradient descent
#model.coef_   -> Intercept   and model.intercept_    -> Slope

print(model.intercept_)
print(model.coef_)
#Evaluate the model using MSE
mse = mean_squared_error(x,y)
print("MSE : ",math.sqrt(mse))
print("R squared value :",model.score(x,y))

#Visualise the model
plt.scatter(x,y,color='green')
plt.plot(x,model.predict(x),color="red")    #model.predict is used to predict the value of x
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()
