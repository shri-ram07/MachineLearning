from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np


data = pd.read_csv("C:\ML\PythonMachineLearning\Datasets\Datasets\credit_data.csv")
"""#data Analysis using co-relation metrix
print(data.corr())"""

#data preprocessing
feature = data[["income","age","loan"]]
target = data.default

x= np.array(feature).reshape(-1,3)    #remove serial column and make it 3 dimensional
y=np.array(target)


#model selection
model = LogisticRegression()
prediction=cross_validate(model,x,y,cv=2)   #cv means fold
print(np.mean(prediction['test_score']))     #Return dictionary with test_score key which is array of test score in each fold





