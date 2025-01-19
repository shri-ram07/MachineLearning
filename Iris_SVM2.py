from sklearn import svm , datasets
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV



#This data sets consists of 3 different types of irises’ (Setosa 0, Versicolour 1, and Virginica 2) petal and sepal length, stored in a 150x4 numpy.ndarray
#The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width.

data = datasets.load_iris()

#Data Splitting
train_x , test_x , train_y , test_y = train_test_split(data.data,data.target,test_size=0.3)

#Model Selection
model = svm.SVC()

#Grid Search is a technique used to tune hyperparameters in machine learning models.
#It exhaustively searches through a specified hyperparameter space
 #to find the combination that gives the best performance for a given model.
parameter_grid = {
    'C':[0.1,1,5,10,20,30,40,50,60,70,100,200],
    'gamma':[1,0.1,0.01,0.001],
    'kernel':['rbf','poly','sigmoid']

}
"""
he refit parameter in GridSearchCV is pretty handy. By setting refit=True,
 which is the default, GridSearchCV will automatically refit the best 
 model on the entire dataset once the best hyperparameters have been 
 found. Here’s why it’s useful
 """
grid = GridSearchCV(model,parameter_grid,refit=True)
grid.fit(train_x,train_y)
predictions = grid.predict(test_x)

print(grid.best_estimator_)   #grid.best_estimator_ used to see the best combination [SVC(C=20, gamma=0.01) for this model]

#Evaluate The model
print("Confusion Metrix : ",confusion_matrix(test_y,predictions))
print("Accuracy Score : ",accuracy_score(test_y,predictions))
"""
Confusion Metrix :  [[14  0  0]
 [ 0 14  0]
 [ 0  0 17]]
Accuracy Score :  1.0
"""