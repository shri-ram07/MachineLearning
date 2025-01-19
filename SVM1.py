import numpy as np
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score

#Feature of class blue
x1 = np.array([0.3,0.5,1,1.4,1.7,2])
y1 = np.array([1,4.5,2.3,1.9,8.9,4.1])

#Features of class red
x2 = np.array([3.3,3.5,4,4.4,5.7,6])
y2 = np.array([7,1.5,6.3,1.9,2.9,7.1])

#main table
x=[
    [0.3  ,  1],
    [0.5  ,  4.5],
    [1    ,  2.3],
    [1.4  ,  1.9],
    [1.7  ,  8.9],
    [2    ,  4.1],
    [3.3  ,  7],
    [3.5  ,  1.5],
    [4    ,  6.3],
    [4.4  ,  1.9],
    [5.7  ,  2.9],
    [6    ,  7.1],
]

y=[0,0,0,0,0,0,1,1,1,1,1,1]

#Data splitting
train_x , test_x , train_y , test_y = train_test_split(x,y,test_size=0.3)


#Data Visualisation

plt.scatter(x1,y1 , color="blue")
plt.scatter(x2,y2 , color="red")

#	Important parameters for SVC: gamma and C
#		gamma -> defines how far the influence of a single training example reaches
#					Low value: influence reaches far      High value: influence reaches close
#
#		C -> trades off hyperplane surface simplicity + training examples missclassifications
#					Low value: simple/smooth hyperplane surface
#					High value: all training examples classified correctly but complex surface
#Model Selection
model = svm.SVC(C=1000)
model.fit(train_x,train_y)

#Prediction
pred = model.predict(test_x)

#Evalutaion
print("Confusion Metrix : ",confusion_matrix(test_y,pred))
print("Accuracy Score: ",accuracy_score(test_y,pred))
plot_decision_regions(np.array(train_x),np.array(train_y),clf=model)

plt.show()