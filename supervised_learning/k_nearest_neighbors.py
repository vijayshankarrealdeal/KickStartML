import numpy as np
from kickstartml.utils import eucliden_distance

class KNN():
    def __init__(self,k = 5):
        self.k = k
    
    def _vote(self,neighbor_labels):
        counts = np.bicount(neighbor_labels.astype('int'))
        return counts.argmax()
        
    def predict(self,X_test,X_train,y_train):
        y_predict = np.array(X_test.shape[0])
        for i,test_sample in enumerate(X_test):
            idx = np.argsort([eucliden_distance(test_sample,x) for x in X_train])[:self.k]
            k_nearest = np.array([y_train[i] for i in idx]) 
            y_predict = self._vote(k_nearest)
        
        return y_predict
"""Test OF Algo"""

import pandas as pd
dataset = pd.read_csv('data/TempLinkoping2016.txt',delimiter = "\t")

X = dataset.iloc[:,:1].values
y = dataset.iloc[:,1:2].values


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.3)

knn = KNN()

knn.predict(X_test,X_train,y_train)