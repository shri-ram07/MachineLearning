from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn import preprocessing


data = pd.read_csv("C:\ML\PythonMachineLearning\Datasets\Datasets\credit_data.csv")
"""#data Analysis using co-relation metrix
print(data.corr())"""

#data preprocessing
feature = data[["income","age","loan"]]
target = data.default
x=np.array(feature).reshape(-1,3)
y=np.array(target)
"""
X before preprocessing
[[6.61559251e+04 5.90170151e+01 8.10653213e+03]
 [3.44151540e+04 4.81171531e+01 6.56474502e+03]
 [5.73171701e+04 6.31080495e+01 8.02095330e+03]
 ...
 [4.43114493e+04 2.80171669e+01 5.52278669e+03]
 [4.37560566e+04 6.39717958e+01 1.62272260e+03]
 [6.94365796e+04 5.61526170e+01 7.37883360e+03]]
 """


#DataPreprocessing
x=preprocessing.MinMaxScaler().fit_transform(x)   #(Normalise the data between 0 and 1 )
"""x after preprocessing
[[0.9231759  0.89209175 0.58883739]
 [0.28812165 0.65470788 0.47682695]
 [0.74633429 0.9811888  0.58262011]
 ...
 [0.48612202 0.21695807 0.40112895]
 [0.47500998 1.         0.1177903 ]
 [0.98881367 0.82970913 0.53597028]]

"""
#train_test_split   70% -> train
feature_train , feature_test ,target_train , target_test=train_test_split(x , y,test_size=0.3)



#model selection
model=KNeighborsClassifier(n_neighbors=20)
model.fit(feature_train,target_train)
prediction = model.predict(feature_test)




#Prediction



#Evaluate of model
print("Confusion Metric : ",confusion_matrix(target_test,prediction))
print("Accuracy Score: ", accuracy_score(target_test,prediction))


"""
Score before preprocessing
Confusion Metric :  [[502   9]
 [ 81   8]]
Accuracy Score:  0.85
"""

"""
Score after preprocessing
Confusion Metric :  [[512   8]
 [ 12  68]]
Accuracy Score:  0.9666666666666667
"""



"""print(model.coef_)
[[-2.36654743e-04 -3.38776663e-01  1.71319503e-03]]

print(model.intercept_)
[9.5963452]
"""



"""
Confusion Metric :  [[498  15]
                    [ 17  70]]
Accuracy Score:  0.9466666666666667
 """
