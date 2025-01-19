from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd


data = pd.read_csv("C:\ML\PythonMachineLearning\Datasets\Datasets\credit_data.csv")
"""#data Analysis using co-relation metrix
print(data.corr())"""

#data preprocessing
feature = data[["income","age","loan"]]
target = data.default


#train_test_split   70% -> train
feature_train , feature_test ,target_train , target_test=train_test_split(feature , target,test_size=0.3)


#model selection
model = LogisticRegression()
model.fit(feature_train,target_train)


#Prediction
prediction=model.predict(feature_test)


#Evaluate of model
print("Confusion Metric : ",confusion_matrix(target_test,prediction))
print("Accuracy Score: ", accuracy_score(target_test,prediction))
"""
print(model.coef_)
[[-2.36654743e-04 -3.38776663e-01  1.71319503e-03]]

print(model.intercept_)
[9.5963452]

"""


"""
Confusion Metric :  [[498  15]
                    [ 17  70]]
Accuracy Score:  0.9466666666666667
 """
