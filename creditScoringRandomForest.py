import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate


data = pd.read_csv(r"C:\Users\rauna\OneDrive\Desktop\ML\PythonMachineLearning\Datasets\Datasets\credit_data.csv")
features = data[["income","age","loan"]]
target = data.default

X = np.array(features).reshape(-1,3)
Y = np.array(target)


model = RandomForestClassifier()
predicted = cross_validate(model,X,Y,cv=10)
print(np.mean(predicted['test_score']))
