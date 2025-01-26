from sklearn.feature_extraction.text import TfidfVectorizer ,CountVectorizer , TfidfTransformer

vec = TfidfVectorizer()
cnt_vec =  CountVectorizer()
new= TfidfTransformer()
doc_term_matrix = new.fit_transform([
    ['My Name is shri ram '
    ,'I love eating healthy food',
    'Ram is a good boy',
    'My family love eating fruits']
])
print(doc_term_matrix.toarray())