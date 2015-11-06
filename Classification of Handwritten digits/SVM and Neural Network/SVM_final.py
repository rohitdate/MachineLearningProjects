# -*- coding: utf-8 -*-
"""
Created on Mon May 11 22:07:25 2015
final svm
@author: Aamir
"""
from sklearn import svm, metrics
from numpy import *
from sklearn import *
from sklearn.metrics import accuracy_score
import csv
import sys

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

test=[]
f3=open("test.csv","r")
datareader3=csv.reader(f3)

for row in datareader3:
    intlist=[]
    intlist = [round((int(x)/255.),2) for x in row]
    test.append(intlist)
    
print "Applying a learning algorithm..."

clf=svm.SVC(C=64, cache_size=1000, class_weight=None, coef0=0.0, degree=3,
  gamma=0.03125, kernel='rbf', max_iter=-1, probability=False,
  random_state=None, shrinking=True, tol=0.001, verbose=False)
  
  #1) 0.03125  C=100000.0
  #2) 0.03125  C=64
  #3) c 1000 -g 0.05
  #4)C=2.8, gamma=.0073

clf.fit(train,train_labels)

predicted = clf.predict(test)
#print predicted[0:100]

with open('SVM_results.csv','w')as f:
    
    writer=csv.writer(f,lineterminator='\n')
    writer.writerow(["ImageID","Label"])
    i=1
    for val in predicted:
        writer.writerow([i,val])
        i+=1