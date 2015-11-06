# -*- coding: utf-8 -*-

import pandas as pd
import time
from sklearn import metrics
from sklearn import decomposition
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import csv

PCA_COMPONENTS = 100

train = pd.read_csv("train1.csv")
features = train.columns[1:]
X = train[features]
y = train['label']

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state=0)

pca=decomposition.PCA(n_components=PCA_COMPONENTS).fit(X_train)
X_train_reduced = pca.transform(X_train)

accuracy_dict={}
values_dict={}

for k in [3,5,10,20]:
    start_time=time.time()
    print "FOR K= ",k
    clf=KNeighborsClassifier(k)
    clf.fit(X_train_reduced, y_train)

    X_test_reduced = pca.transform(X_test)

    y_pred=clf.predict(X_test_reduced)

    values_dict[k]=y_pred

    print classification_report(y_test, y_pred)
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test,y_pred))

    acc = accuracy_score(y_test, y_pred)
    
    accuracy_dict[k]=acc

    print("\n")
    print("Accuracy:%f" %(acc*100))

    print("Runtime:" )
    print round((time.time()-start_time),2)," seconds"


max_key=max(accuracy_dict, key=accuracy_dict.get)

with open('KNN_results.csv','w')as f:
    
    writer=csv.writer(f,lineterminator='\n')
    writer.writerow(["ImageID","Label"])
    i=1
    for val in values_dict[max_key]:
        writer.writerow([i,val])
        i+=1