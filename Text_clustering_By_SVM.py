
from sklearn import svm , datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.feature_extraction.text import CountVectorizer,  TfidfTransformer



categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']
data = fetch_20newsgroups(subset='train',categories=categories,shuffle=True , random_state=True)
train_x , test_x , train_y , test_y = train_test_split(data.data,data.target,test_size=0.3)

cnt_vec = CountVectorizer()
train_x = cnt_vec.fit_transform(train_x)

tfidf = TfidfTransformer()
train_x = tfidf.fit_transform(train_x)
test_x = cnt_vec.transform(test_x)
test_x = tfidf.transform(test_x)
model = svm.SVC(C=1000000 )
model.fit(train_x,train_y)


predictions = model.predict(test_x)

#Evaluate The model
print("Confusion Metrix : ",confusion_matrix(test_y,predictions))
print("Accuracy Score : ",accuracy_score(test_y,predictions))

