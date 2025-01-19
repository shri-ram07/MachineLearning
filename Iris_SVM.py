from sklearn import svm , datasets
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split

#This data sets consists of 3 different types of irises’ (Setosa 0, Versicolour 1, and Virginica 2) petal and sepal length, stored in a 150x4 numpy.ndarray
#The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width.

data = datasets.load_iris()

#Data Splitting
train_x , test_x , train_y , test_y = train_test_split(data.data,data.target,test_size=0.3)

#Model Selection
model = svm.SVC()
model.fit(train_x,train_y)


predictions = model.predict(test_x)

#Evaluate The model
print("Confusion Metrix : ",confusion_matrix(test_y,predictions))
print("Accuracy Score : ",accuracy_score(test_y,predictions))
"""
Confusion Metrix :  [[14  0  0]
 [ 0 14  0]
 [ 0  0 17]]
Accuracy Score :  1.0
"""