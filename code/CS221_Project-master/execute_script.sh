#!/bin/bash

UCF_DIR="/media/sf_DON/DATASET/UCF-101/"
TRAIN_LIST="/media/sf_DON/DATASET/UCF-101/ucfTrainTestlist/trainlist01.txt"
GMM_OUT="./UCF101_Fishers/gmm_list"

python gmm.py 120 $UCF_DIR $TRAIN_LIST $GMM_OUT --pca

trainlist01="/media/sf_DON/DATASET/UCF-101/ucfTrainTestlist/trainlist01.txt"
testlist01="/media/sf_DON/DATASET/UCF-101/ucfTrainTestlist/testlist01.txt"

training_output="./UCF101_Fishers/train"
testing_output="./UCF101_Fishers/test"

python computeFVs.py $UCF_DIR $trainlist01 $training_output $GMM_OUT
python computeFVs.py $UCF_DIR $testlist01 $testing_output $GMM_OUT

#CLASS_INDEX="/Users/Bryan/CS/CS_Research/data/class_attributes_UCF101/Class_Index.txt"
#CLASS_INDEX_OUT="./class_index"
###python compute_UCF101_class_index.py $CLASS_INDEX $CLASS_INDEX_OUT
#
#python classify_experiment.py
