from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer()

#Fitiing the text
"""fit_transform(raw_documents, y=None)
Learn vocabulary and idf, return document-term matrix.
This is equivalent to fit followed by transform, but more efficiently implemented."""

doc_term_matrix = vec.fit_transform([
    'My Name is shri ram '
    ,'I love eating healthy food',
    'Ram is a good boy',
    'My family love eating fruits'
])


"""
print(doc_term_matrix.toarray())         # It shows a document term matrix

[[0.         0.         0.         0.         0.         0.
  0.         0.40104275 0.         0.40104275 0.50867187 0.40104275
  0.50867187]
 [0.         0.43779123 0.         0.55528266 0.         0.
  0.55528266 0.         0.43779123 0.         0.         0.
  0.        ]
 [0.55528266 0.         0.         0.         0.         0.55528266
  0.         0.43779123 0.         0.         0.         0.43779123
  0.        ]
 [0.         0.40104275 0.50867187 0.         0.50867187 0.
  0.         0.         0.40104275 0.40104275 0.         0.
  0.        ]]

"""


#Print the similarity matrix
print((doc_term_matrix*doc_term_matrix.T).toarray())
"""
[[1.         0.         0.351146   0.16083528]
 [0.         1.         0.         0.351146  ]
 [0.351146   0.         1.         0.        ]
 [0.16083528 0.351146   0.         1.        ]]
"""