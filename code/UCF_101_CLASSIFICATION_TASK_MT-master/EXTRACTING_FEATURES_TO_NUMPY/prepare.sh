#!/bin/bash


path_to_file_train_list="./HOG_NPY/train_per_class_npy_file_list.txt"
prefix_train_filename="trainall"

path_to_file_test_list="./HOG_NPY/test_per_class_npy_file_list.txt"
prefix_test_filename="test"

output_path="../data/"

testmode=0

python prepare_features.py $path_to_file_train_list $prefix_train_filename $path_to_file_test_list $prefix_test_filename $output_path $testmode > output_prepare_features.txt
