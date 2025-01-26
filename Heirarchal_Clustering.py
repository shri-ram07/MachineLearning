from scipy.cluster.hierarchy import dendrogram,linkage
from matplotlib import pyplot as plt
import numpy as np
x = np.array([[1, 1], [1.5, 1], [3, 3], [4, 4], [3, 3.5], [3.5, 4]])

linkage = linkage(x,"single")
dendrogram = dendrogram(linkage,truncate_mode="none")

plt.title("Hierarchical Clustering")
plt.show()