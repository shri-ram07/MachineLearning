import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#Feature of class blue
x1 = np.array([0.3,0.5,1,1.4,1.7,2])
y1 = np.array([1,4.5,2.3,1.9,8.9,4.1])

#Features of class red
x2 = np.array([3.3,3.5,4,4.4,5.7,6])
y2 = np.array([7,1.5,6.3,1.9,2.9,7.1])

#main table
x=[
    [0.3  ,  1],
    [0.5  ,  4.5],
    [1    ,  2.3],
    [1.4  ,  1.9],
    [1.7  ,  8.9],
    [2    ,  4.1],
    [3.3  ,  7],
    [3.5  ,  1.5],
    [4    ,  6.3],
    [4.4  ,  1.9],
    [5.7  ,  2.9],
    [6    ,  7.1],
]

y=[0,0,0,0,0,0,1,1,1,1,1,1]

#Visualise the data
plt.scatter(x1,y1,color="blue")
plt.scatter(x2,y2,color="red")
plt.scatter(3,5,color='green')



#Model Selection
model = KNeighborsClassifier(n_neighbors=3)  #n_neighbors = k
model.fit(x,y)

print(model.predict([[3,5]]))







plt.show()



