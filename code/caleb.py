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
import numpy as np

import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #To suppress warnings about CPU instruction set

import re
import numpy as np
#for appleeyemakeup
a=[]
data0=[]
count=-1
for i in range(1,26):
      for j in range(1,5):
            count=count+1
            if(i<10):
                  #a.append
                  d=open("./UCF_101_CLASSIFICATION_TASK_MT-master/EXTRACTING_FEATURES_TO_NUMPY/HOG_FEA/ApplyEyeMakeup/v_ApplyEyeMakeup_g0"+str(i)+"_c0"+str(j)+".fea","r")
                  a.append(d)
            else:
                  d=open("./UCF_101_CLASSIFICATION_TASK_MT-master/EXTRACTING_FEATURES_TO_NUMPY/HOG_FEA/ApplyEyeMakeup/v_ApplyEyeMakeup_g"+str(i)+"_c0"+str(j)+".fea","r")
                  a.append(d)
            data0.append(a[count].read().split("\t"))
#data=data[:][0:3]
#print(len(data[0]))
#print(len(data[99]))
#print((data[99][43600]))
#jef=data[:][0:43]
#print(len(jef[0]))
b=[]
data1=[]
count=-1
for i in range(1,21):
      for j in range(1,5):
            count=count+1
            if(i<10):
                  d=open("./UCF_101_CLASSIFICATION_TASK_MT-master/EXTRACTING_FEATURES_TO_NUMPY/HOG_FEA/ApplyLipstick/v_ApplyLipstick_g0"+str(i)+"_c0"+str(j)+".fea","r")
                  b.append(d)
            else:
                  d=open("./UCF_101_CLASSIFICATION_TASK_MT-master/EXTRACTING_FEATURES_TO_NUMPY/HOG_FEA/ApplyLipstick/v_ApplyLipstick_g"+str(i)+"_c0"+str(j)+".fea","r")
                  b.append(d)
            data1.append(b[count].read().split("\t"))



c=[]
data2=[]
count=-1
for i in range(1,26):
      for j in range(1,5):
            count=count+1
            if(i<10):
                  d=open("./UCF_101_CLASSIFICATION_TASK_MT-master/EXTRACTING_FEATURES_TO_NUMPY/HOG_FEA/BabyCrawling/v_BabyCrawling_g0"+str(i)+"_c0"+str(j)+".fea","r")
                  c.append(d)
            else:
                  d=open("./UCF_101_CLASSIFICATION_TASK_MT-master/EXTRACTING_FEATURES_TO_NUMPY/HOG_FEA/BabyCrawling/v_BabyCrawling_g"+str(i)+"_c0"+str(j)+".fea","r")
                  c.append(d)
            data2.append(c[count].read().split("\t"))
e=[]
data3=[]
count=-1
for i in range(1,26):
      for j in range(1,5):
            count=count+1
            if(i<10):
                  d=open("./UCF_101_CLASSIFICATION_TASK_MT-master/EXTRACTING_FEATURES_TO_NUMPY/HOG_FEA/BaseballPitch/v_BaseballPitch_g0"+str(i)+"_c0"+str(j)+".fea","r")
                  e.append(d)
            else:
                  d=open("./UCF_101_CLASSIFICATION_TASK_MT-master/EXTRACTING_FEATURES_TO_NUMPY/HOG_FEA/BaseballPitch/v_BaseballPitch_g"+str(i)+"_c0"+str(j)+".fea","r")
                  e.append(d)
            data3.append(e[count].read().split("\t"))
f=[]
data4=[]
count=-1
for i in range(1,26):
      for j in range(1,5):
            count=count+1
            if(i<10):
                  d=open("./UCF_101_CLASSIFICATION_TASK_MT-master/EXTRACTING_FEATURES_TO_NUMPY/HOG_FEA/BenchPress/v_BenchPress_g0"+str(i)+"_c0"+str(j)+".fea","r")
                  f.append(d)
            else:
                  d=open("./UCF_101_CLASSIFICATION_TASK_MT-master/EXTRACTING_FEATURES_TO_NUMPY/HOG_FEA/BenchPress/v_BenchPress_g"+str(i)+"_c0"+str(j)+".fea","r")
                  f.append(d)
            data4.append(f[count].read().split("\t"))


print(len(data0))
print(len(data1))
print(len(data2))
print(len(data3))
print(len(data4))

fdata=np.concatenate((data0, data1,data2,data3,data4), axis=0)
print(len(fdata[0]))





# Creates a list containing 100frames for 480 dataset, each of 436 features
great=np.zeros(((400*100),436))
print(great.shape)

for i in range(0,4):
      for k in range(0,100):
            for l in range(0,100):
                  
                  for j in range(0,436):
                        if((i*10000)<10000):
                              #data0
                              great[(i*10000)+(k*100)+l][j]=float(data0[k][(l*436)+j])
                        elif((i*10000)<20000):
                              #data2
                              great[(i*10000)+(k*100)+l][j]=float(data2[k][(l*436)+j])
                        elif((i*10000)<30000):
                              #data3
                              great[(i*10000)+(k*100)+l][j]=float(data3[k][(l*436)+j])
                        else:
                              #data4
                              great[(i*10000)+(k*100)+l][j]=float(data4[k][(l*436)+j])
print(great[30000][435])

     
#print(arr[99][435])
#creating an  array 
                              
"""                              
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
"""

#print(great[400])
y=np.zeros(400*100)
print(y.shae)
count=-1
for i in range(0,40000):
      if(i%10000==0):
            count=count+1
      y[i]=count

print(y.shape)

neigh = KNeighborsClassifier(n_neighbors=5,algorithm='kd_tree')
#clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(neigh, great, y, cv=5)
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


