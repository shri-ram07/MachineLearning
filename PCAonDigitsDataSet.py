from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn import datasets

digits=datasets.load_digits()

X = digits.data
Y = digits.target

est = PCA(n_components=10)     #transform 64 features into 2 features
X_pca = est.fit_transform(X)

colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']

for i in range(len(colors)):
    px = X_pca[:,0][Y==i]
    py = X_pca[:,1][Y==i]
    plt.scatter(px,py,c=colors[i])
    plt.legend(digits.target_names)

plt.xlabel('First Principle Component')
plt.ylabel('Second Principle Component')

plt.show()
# explained variance shows how much information can be attributed to the principle components
print("Explained variance: %s" % est.explained_variance_ratio_)

plt.show()