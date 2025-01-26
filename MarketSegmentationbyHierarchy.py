import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv("Datasets/Datasets/shopping_data.csv")
#Annual Income (k$),Spending Score (1-100)  ---->features
data = data.iloc[: , 3:5].values

plt.figure(figsize=(10, 10))
plt.title("Market Segmentation Data")
dendogram = dendrogram(linkage(data , method='ward'))
plt.show()

cluster = AgglomerativeClustering(n_clusters=5)
labels = cluster.fit_predict(data)
plt.figure(figsize=(10, 10))
plt.scatter(data[:,0],data[:,1],c=labels,cmap='rainbow')
plt.title("Market Segmentation Data")
plt.xlabel("Income")
plt.ylabel("Expenses Score")
plt.show()