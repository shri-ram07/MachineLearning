import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

x1 = np.array([0, 0.6, 1.1, 1.5, 1.8, 2.5, 3, 3.1, 3.9, 4, 4.9, 5, 5.1])
y1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])

x2 = np.array([3, 3.8, 4.4, 5.2, 5.5, 6.5, 6, 6.1, 6.9, 7, 7.9, 8, 8.1])
y2 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])

x = np.array([[0], [0.6], [1.1], [1.5], [1.8], [2.5], [3], [3.1], [3.9], [4], [4.9], [5], [5.1] , [3], [3.8], [4.4], [5.2], [5.5], [6.5], [6], [6.1], [6.9], [7], [7.9], [8], [8.1]])
y = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])

plt.scatter(x1,y1 , color="red")
plt.scatter(x2,y2,color="blue")
plt.show()

model = LogisticRegression()
model.fit(x,y)

"""
intercept = [-4.50163542]
coefficient = [[1.00401882]]

"""

print(model.predict_proba([[4.4789999999999999]]))    #predict the probability
