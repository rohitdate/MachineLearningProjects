# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:27:58 2015

@author: Kaly
"""
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import *
from sklearn.metrics import classification_report

import csv


train=[]

f1=open("train.csv","r")
datareader=csv.reader(f1)

for row in datareader:
    intlist=[]
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

print "Applying a learning algorithm..."


from nolearn.dbn import DBN
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
clf = DBN(
    [X_train.shape[1], 300, 10],
    learn_rates=0.3,
    learn_rate_decays=0.9,
    epochs=15,
    verbose=1,
    )

clf.fit(X_train, y_train)
acc_nn = clf.score(X_test,y_test)
print "neural network accuracy: ",acc_nn


y_pred = clf.predict(X_test)
print "Classification report:"
print classification_report(y_test, y_pred)
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test,y_pred))

