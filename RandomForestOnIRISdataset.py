from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn import datasets


data = datasets.load_iris()

features = data.data
target = data.target

train_x , test_x ,train_y , test_y = train_test_split(features,target,test_size=0.2)

model = RandomForestClassifier(n_estimators=10000,max_features="sqrt")
model.fit(train_x,train_y)


predictions = model.predict(test_x)

print("confusion metrix : ",confusion_matrix(test_y,predictions))
print("accuracy Score : ",accuracy_score(test_y,predictions))