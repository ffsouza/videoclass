#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:55:24 2017

@author: caleb192
"""



"""
Script to train a basic action classification system.

Trains a One vs. Rest SVM classifier on the fisher vector video outputs.
This script is used to experimentally test different parameter settings for the SVMs.

"""

import os, sys, collections, random, string
import numpy as np
from tempfile import TemporaryFile
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def toDict(videos):
    videos_by_class = dict()
    for video in videos:
        #we assume each of the videos has the following format: 
        # v_BasketballDunk_g02_c02.fisher.npz
        name = string.lower(video.split('.')[1])
        if name not in videos_by_class:
            videos_by_class[name] = []
        videos_by_class[name].append(video)
    return videos_by_class

def limited_input1(input_dict, T):
    vids = []
    for k,v in input_dict.iteritems():
        if len(v) <= T:
            vids.extend(v)
        else:
            vids.extend(random.sample(v,T))
    return vids


class_index_file = "./class_index.npz"
training_output = './data/train'
testing_output = './data/test'

class_index_file_loaded = np.load(class_index_file)
class_index = class_index_file_loaded['class_index'][()]
index_class = class_index_file_loaded['index_class'][()]




training = [filename for filename in os.listdir(training_output) if filename.endswith('.features')]
testing = [filename for filename in os.listdir(testing_output) if filename.endswith('.features')]


training_dict = toDict(training)
testing_dict = toDict(testing)


#GET THE TRAINING AND TESTING DATA.


#X_train_vids, X_test_vids = limited_input(training_dict, testing_dict, 100, 24)
X_train, Y_train = make_FV_matrix(training_dict,training_output, class_index)
X_test, Y_test = make_FV_matrix(testing_dict,testing_output, class_index)

training_PCA = limited_input1(training_dict,1)



#Experiments with PCA
pca_dim = 1000
pca = PCA(n_components=pca_dim)
pca.fit(X_train)
X_train_PCA = pca.transform(X_train)
X_test_PCA = pca.transform(X_test)
estimator = OneVsRestClassifier(LinearSVC(random_state=0, C=100, loss='l1', penalty='l2'))
classifier = estimator.fit(X_train_PCA, Y_train)
metrics = classify_library.metric_scores(classifier, X_test_PCA, Y_test, verbose=True)
print metrics


do_learning_curve = False
if do_learning_curve:
    X_full = np.vstack([X_train_PCA, X_test_PCA])
    Y_full = np.hstack([Y_train, Y_test])
    title= "Learning Curves (Linear SVM, C: %d, loss: %s, penalty: %s, PCA dim: %d)" % (100,'l1','l2',pca_dim)
    cv = cross_validation.ShuffleSplit(X_full.shape[0], n_iter=4,test_size=0.2, random_state=0)
    estimator = OneVsRestClassifier(LinearSVC(random_state=0, C=100, loss='l1', penalty='l2'))
    plot_learning_curve(estimator, title, X_full, Y_full, (0.7, 1.01), cv=cv, n_jobs=1)
    plt.show()

