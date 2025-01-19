"""
The 20 Newsgroups dataset is a popular collection of approximately 18,000 newsgroup posts across 20 different topics
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
"""
MultinomialNB for text data and GaussianNB for numerical data that follows a normal distribution1
"""


categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
]

data = fetch_20newsgroups(subset="train",categories=categories , shuffle=True , random_state=True)   #Return a dictionary like data called Bunch
"""
dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR'])
"""

# Count occurrences of words
cnt_vec =  CountVectorizer()
x_train_count = cnt_vec.fit_transform(data.data)


#we transform the word  into tfidf
tfidf = TfidfTransformer()
x_train = tfidf.fit_transform(x_train_count)  #(0, 1): The value at row 0, column 1 is 0.56.
print(x_train)

#Model Selection
model =  MultinomialNB().fit(x_train , data.target)


#Prediction
test = ['My Computer is So fast ' , 'My father i a software engineer' , 'he has windows licence key']
x_test_count = cnt_vec.transform(test)
x_test_tfidf = tfidf.transform(x_test_count)


prediction = model.predict(x_test_tfidf)

for x,i in zip(test ,prediction):
 cat = data.target_names[i]
 print(f"{x}--------------------------->{cat}")

