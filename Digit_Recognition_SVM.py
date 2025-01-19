from sklearn import svm,datasets
from sklearn.metrics import confusion_matrix , accuracy_score





"""
This dataset is made up of 1797 8x8 images. 
Every point on metrix represent pixel intensity
Each image, like the one shown below, is of a hand-written digit. 
In order to utilize an 8x8 figure like this, 
we’d have to first transform it into a feature vector with length 64 (features)"""
data = datasets.load_digits()
#features = data.images   , Target = data.target

#Data Preprocessing
"""
The enumerate() function in Python is a built-in function that adds
 a counter to an iterable and returns it as an enumerate object.
  This function is particularly useful when you need to loop 
  over an iterable and keep track of the index of each item."""


# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(data.images)
data2 = data.images.reshape((n_samples, -1))      # Convert it to 1 d array
"""
[[ 0.  0.  5. ...  0.  0.  0.]
 [ 0.  0.  0. ... 10.  0.  0.]
 [ 0.  0.  0. ... 16.  9.  0.]
 ...
 [ 0.  0.  1. ...  6.  0.  0.]
 [ 0.  0.  2. ... 12.  0.  0.]
 [ 0.  0. 10. ... 12.  1.  0.]]"""


#New Splitting Technique
train_test = int(len(data.images)*0.75)

#Model Selection
model = svm.SVC(gamma=0.001)
model.fit(data2[:train_test],data.target[:train_test])


#Now predict the value
expected = data.target[train_test:]
predicted = model.predict(data2[train_test:])

#Evaluate the model
print("Confusion Metrix : ",confusion_matrix(expected,predicted))
print("Accuracy Score : ",accuracy_score(expected,predicted))



