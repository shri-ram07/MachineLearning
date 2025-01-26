from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml     #digit set
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



mnist = fetch_openml('mnist_784', version=1)

x = mnist.data
y = mnist.target

print(x.shape)