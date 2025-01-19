import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#Data Preprocessing

data = pd.read_csv("C:\ML\PythonMachineLearning\Datasets\Datasets\credit_data.csv")
feature = data[["income","age","loan"]]
target = data.default

#Data Splitting
feature_train , feature_test ,target_train , target_test=train_test_split(feature , target,test_size=0.3)

#Model Selection
model = GaussianNB()

#Model Training
model.fit(feature_train,target_train)

#Prediction
prediction = model.predict(feature_test)

#Evaluation
print("confusionMetrix : ",confusion_matrix(prediction,target_test))
print("Accuracy : ",accuracy_score(prediction,target_test))
"""
confusionMetrix :  [[510  33]
                    [9  48]]
Accuracy :  0.93

"""
