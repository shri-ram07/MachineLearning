from PIL.EpsImagePlugin import split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



data = pd.read_csv('C:\ML\PythonMachineLearning\Datasets\Datasets\SPAM text message 20170820 - Data.csv')
features = data['Message']

target = data.Category
x=np.array(features)
y=np.array(target)
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3)

cnt = CountVectorizer()
x_train = cnt.fit_transform(x_train)

tfidf = TfidfTransformer()
x_train = tfidf.fit_transform(x_train)



#Model Selection
model = MultinomialNB().fit(x_train,y_train)


test =cnt.transform(x_test)
test = tfidf.transform(test)

prediction = model.predict(test)
print("Confusion Metrix : ",confusion_matrix(y_test,prediction))
print("Accuracy Score : ",accuracy_score(y_test,prediction))


"""
Confusion Metrix :  [[1446    0]
 [  71  155]]
Accuracy Score :  0.9575358851674641
"""

