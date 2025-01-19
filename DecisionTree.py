import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import datasets
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.model_selection import GridSearchCV


iris_dataset = datasets.load_iris()

features  = iris_dataset.data
target = iris_dataset.target

train_x , test_x , train_y , test_y = train_test_split(features,target,test_size=0.2)

#with grid search you can find an optimal parameter "Parameter Tuning"
param_grid = {'max_depth':np.arange(1,10)}

#it will test on all above possible param and train on best param
model = GridSearchCV(DecisionTreeClassifier(),param_grid)
model.fit(train_x,train_y)

print("Best Parameter : ",model.best_params_)

prediction = model.predict(test_x)
print(confusion_matrix(test_y,prediction))
print(accuracy_score(test_y,prediction))