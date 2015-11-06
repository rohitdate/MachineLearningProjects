# -*- coding: utf-8 -*-
"""
Created on Mon May 11 19:09:57 2015
grid search
@author: Aamir
"""

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
import numpy as np
import csv
from sklearn import *
 
def main():
      
    train=[]

    f1=open("train.csv","r")
    datareader=csv.reader(f1)

    for row in datareader:
        intlist=[]
        #scaling
        intlist = [ round((int(x)/255.),2) for x in row]
        train.append(intlist)


    train_labels=[]

    f2=open("train_labels.csv","r")
    datareader2=csv.reader(f2)

    for row in datareader2:
        labellist=[]
        labellist=[float(x) for x in row]
        train_labels.extend(labellist)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train,train_labels,test_size=0.2,random_state=0)
   
    svm = SVC(cache_size=1000, kernel='rbf')
    
    C_range = 10. ** np.arange(5,10)
    gamma_range = 2. ** np.arange(-5, -1)

    parameters = dict(gamma=gamma_range, C=C_range)
 
    print("grid search")
    grid = GridSearchCV(svm, parameters, cv=StratifiedKFold(y_train, 5), verbose=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    print("predicting")
    print "score: ", grid.score(X_test, y_test)
    print("The best classifier is: ", grid.best_estimator_)
    
if __name__ == "__main__":
    main()