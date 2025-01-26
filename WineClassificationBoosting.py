from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.model_selection import GridSearchCV , train_test_split
import pandas as pd
from sklearn import preprocessing
import numpy as np


def is_tasty(quality):
    if quality>=7:
        return 1
    else:
        return 0

data = pd.read_csv(r"Datasets/Datasets/wine.csv",sep=";")

features = data[
    ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
     "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]

data['tasty'] = data["quality"].apply(is_tasty)

target = data["tasty"]


X = np.array(features).reshape(-1,11)
Y = np.array(target)

X = preprocessing.MinMaxScaler().fit_transform(X)

param = {
    'n_estimators':[10,50,100,1000,10000],
    'learning_rate':[0.01,0.05,0.3,1],
}
x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.2)
model = GridSearchCV(AdaBoostClassifier(),param,cv=10)
model.fit(x_train,y_train)
predicted = model.predict(x_test)
print("Confusion Metrix : ",confusion_matrix(predicted , y_test))
print("Accuracy Score : ",accuracy_score(predicted , y_test))
print("Best Parameters are : ",model.best_params_)
"""Confusion Metrix :  [[715 127]
 [ 48  90]]
Accuracy Score :  0.8214285714285714
Best Parameters are :  {'learning_rate': 1, 'n_estimators': 10000}"""