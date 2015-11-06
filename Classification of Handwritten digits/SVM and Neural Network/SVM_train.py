# -*- coding: utf-8 -*-
"""
Created on Wed May 06 21:13:17 2015

@author: Aamir
"""


from sklearn.datasets import fetch_mldata
from sklearn import svm, metrics
from numpy import *
from sklearn import *
from sklearn.metrics import accuracy_score
import csv

"""mnist = fetch_mldata('MNIST original')
#print (mnist.data[0])

print("Scaling")
X_all, y_all = mnist.data/255., mnist.target
#print X_all[0]

shuffle=random.permutation(arange(X_all.shape[0]))
X_all,y_all=X_all[shuffle],y_all[shuffle]

X_train,y_train = X_all[:60000,:], y_all[:60000]
X_test, y_test = X_all[60000:,:],y_all[60000:]"""

train=[]

f1=open("train.csv","r")
datareader=csv.reader(f1)

for row in datareader:
    intlist=[]
    intlist = [ round((int(x)/255.),2) for x in row]
    train.append(intlist)
#print train[0:2]

train_labels=[]

f2=open("train_labels.csv","r")
datareader2=csv.reader(f2)

for row in datareader2:
    labellist=[]
    labellist=[float(x) for x in row]
    train_labels.extend(labellist)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(train,train_labels,test_size=0.2,random_state=0)

print "Applying a learning algorithm..."

clf=svm.SVC(C=100000.0, cache_size=1000, class_weight=None, coef0=0.0, degree=3,
  gamma=0.03125, kernel='rbf', max_iter=-1, probability=False,
  random_state=None, shrinking=True, tol=0.001, verbose=False)

clf.fit(X_train,y_train)

#scores=cross_validation.cross_val_score(clf,X_all,y_all,cv=2)
#print scores
print 

expected=(y_test)
predicted = clf.predict(X_test)

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

print ("Accuracy",accuracy_score(expected,predicted))
