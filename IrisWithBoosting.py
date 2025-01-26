from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn import datasets



data = datasets.load_iris()

X = data.data
Y = data.target

features_train , features_test , target_train , target_test = train_test_split(X,Y,test_size=0.2)

model = AdaBoostClassifier(n_estimators=10000 , learning_rate=1 , random_state=123 )
model.fit(features_train,target_train)

predicted = model.predict(features_test)
print("Confusion Metrix : ", confusion_matrix(predicted,target_test))
print("Accuracy Score  : ", accuracy_score(predicted,target_test))