# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:33:01 2015

@author: Kaly
"""

import pandas as pd
#import numpy as np
from sklearn import cross_validation
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from nolearn.dbn import DBN
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics


train = pd.read_csv("train1.csv")
features = train.columns[1:]
X = train[features]
y = train['label']
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X/255.,y,test_size=0.2,random_state=0)


clf_sgd = SGDClassifier(learning_rate='optimal', loss='hinge', n_iter=20, shuffle=True)
clf_sgd.fit(X_train, y_train)
y_pred_sgd = clf_sgd.predict(X_test)
acc_sgd = accuracy_score(y_test, y_pred_sgd)
print "stochastic gradient descent accuracy: ",acc_sgd

print classification_report(y_test, y_pred_sgd)
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test,y_pred_sgd))

