from sklearn.cluster import DBSCAN , KMeans
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

x , y = datasets.make_moons(n_samples=1500, noise=0.05,random_state=3)
x1 = x[:,0]
x2 = x[:,1]



est = DBSCAN(eps=.3)
est.fit(x)
y_pred = est.labels_.astype(int)
print(y_pred)
colors = np.array(["red","green"])
plt.scatter(x1,x2,s=5,color=colors[y_pred])
plt.show()


#by kmeans

est2 = KMeans(n_clusters=2, random_state=3)
est2.fit(x)
y_pred2 = est2.labels_.astype(int)
plt.scatter(x1,x2,s=5,color=colors[y_pred2])
plt.show()

