#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:07:47 2018

@author: karunya
"""
import theano
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import string
import time
from sklearn.model_selection import cross_val_score
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import string
import time
import csv
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import pandas as pd
import seaborn as sns
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #To suppress warnings about CPU instruction set
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #To suppress warnings about CPU instruction set

import re
import numpy as np
#for appleeyemakeup
f=open("./v_ApplyEyeMakeup_g01_c01.fea","r")
a=open("./v_ApplyLipstick_g01_c01.fea","r")
b=open("./v_Archery_g01_c01.fea","r")
c=open("./v_BabyCrawling_g01_c01.fea","r")
d=open("./v_BalanceBeam_g01_c01.fea","r")
print(f.read())
#data=f.read().split("\n")
data=f.read().split("\t")
data1=a.read().split("\t")
data2=b.read().split("\t")
data3=c.read().split("\t")
data4=d.read().split("\t")
#ans=re.split("\t \n",f)
#data.trim("\n")
len(data)
len(data1)
len(data2)
len(data3)
len(data4)

#type(data)
#print (data)
data=data[0:43600]
data1=data1[0:43600]
data2=data2[0:43600]
data3=data3[0:43600]
data4=data4[0:43600]

print(data[0])

# Creates a list containing 100frames, each of 436 features
arr=numpy.zeros((100,436))
print(arr.shape)
#print(arr[99][435])
#creating an  array 
for r in range(0,100):
    for c in range(0,436):
        arr[r][c]=float(data[(r*436)+c])

arr1=numpy.zeros((100,436))
print(arr1.shape)
#print(arr[99][435])
#creating an  array 
for r in range(0,100):
    for c in range(0,436):
        arr1[r][c]=float(data1[(r*436)+c])

arr2=numpy.zeros((100,436))
print(arr2.shape)
#print(arr[99][435])
#creating an  array 
for r in range(0,100):
    for c in range(0,436):
        arr2[r][c]=float(data2[(r*436)+c])

arr3=numpy.zeros((100,436))
print(arr3.shape)
#print(arr[99][435])
#creating an  array 
for r in range(0,100):
    for c in range(0,436):
        arr3[r][c]=float(data3[(r*436)+c])
        
arr4=numpy.zeros((100,436))
print(arr4.shape)
#print(arr[99][435])
#creating an  array 
for r in range(0,100):
    for c in range(0,436):
        arr4[r][c]=float(data4[(r*436)+c])
        
fi=np.concatenate((arr, arr1,arr2,arr3,arr4), axis=0)
print(fi.shape)
fi=fi[:,1:436]

y=np.zeros(500)
print(y)
count=-1
for i in range(0,500):
      if(i%100==0):
            count=count+1
      y[i]=count

print(y.shape)

neigh = KNeighborsClassifier(n_neighbors=5,algorithm='kd_tree')
#clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(neigh, fi, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


clf = svm.SVC()
caleb = cross_val_score(clf, fi, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (caleb.mean(), caleb.std() * 2))

qa=QuadraticDiscriminantAnalysis()
jef = cross_val_score(qa, fi, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (jef.mean(), jef.std() * 2))
qa.fit(fi, y) 


jeffin= XGBClassifier()
jeff = cross_val_score(jeffin, fi, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (jeff.mean(), jeff.std() * 2))


