#!/bin/bash

#Protocol buffers
export PATH="/home/jjorge/protobuf/bin:$PATH"

#Eigen lib
export LD_LIBRARY_PATH=/home/jjorge/eigen/include/eigen3:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/jjorge/deepnet-master/eigenmat:$LD_LIBRARY_PATH

#Deepenet
export PYTHONPATH=/home/jjorge/deepnet-master/
export PYTHONPATH=/home/jjorge/deepnet-master/eigenmat:$PYTHONPATH
export PYTHONPATH=/home/jjorge/protobuf/PY_LIB:$PYTHONPATH
export PYTHONPATH=/home/common/local/lib/python2.7/site-packages:$PYTHONPATH

deepnet_root="/home/jjorge/deepnet-master/deepnet"
trainer_root=$deepnet_root/trainer.py
extract_rep_root=$deepnet_root/extract_neural_net_representation.py

#Program
train_deepnet="python $trainer_root"
extract_rep_deepnet="python $extract_rep_root"
evaluate_deepnet='python evaluate.py'

#Test 

test_labels_pattern="/home/jjorge/HOG_features/features_data/test/test_labels_*.npy"
test_ids_pattern="/home/jjorge/HOG_features/features_data/test/test_ids_*.npy"

#Conf. files
train_conf="conf/train.pbtxt"
eval_conf="conf/eval.pbtxt"
trainall_conf="conf/trainall.pbtxt"

timestamp=$(date +%s)

#Modelo
#for MODELNAME in model_1_layer_128_relu model_1_layer_256_relu model_1_layer_512_relu model_1_layer_1024_relu model_1_layer_2048_relu;
for MODELNAME in model_1_layer_2048_relu;
do
 #Output
 output_file=random_1000_results_0.8_wn_3_20_ep_"$MODELNAME"_"$timestamp".txt

 rm -r train_with_validation/models
 rm -r trainall/models
 rm -r test

 # Train Neural Net 
 # stop criterion: number of steps
 echo "Phase 1==========="
 echo "=================="

 echo "Start training"
 echo "with $MODELNAME"
 start_time=`date +%s`
 ${train_deepnet} train_with_validation/$MODELNAME.pbtxt "$train_conf" "$eval_conf"|| exit 1
 end_time=`date +%s`

 echo execution time was `expr $end_time - $start_time` s.

 # stop criterion: cross entropy error
 # le pasamos el mejor modelo del valid
 echo "Phase 2==========="
 echo "=================="
 echo "Classifier train_full and test"
 echo "with $MODELNAME"

 start_time=`date +%s`
 ${train_deepnet} trainall/$MODELNAME"_trainall".pbtxt "$trainall_conf" "$eval_conf" train_with_validation/models/model_BEST|| exit 1
 end_time=`date +%s`

 echo execution time was `expr $end_time - $start_time` s.

 # EVALUATION
 echo "Phase 3==========="
 echo "=================="
 echo "Extract representation and evaluate"
 echo "with $MODELNAME"

 ${extract_rep_deepnet} trainall/models/model_LAST "$eval_conf" . 1 1 test output_layer

 ${evaluate_deepnet} "$test_labels_pattern" "$test_ids_pattern" test/output_layer-00001-of-00001.npy 0 > "$output_file"

 mv test test_1000_0.8_wn_3_20_ep_"$MODELNAME"
 mv train_with_validation/models train_with_validation/models_1000_0.8_wn_3_20_ep_"$MODELNAME"
 mv trainall/models trainall/models_1000_0.8_wn_3_20_ep_"$MODELNAME"


 echo "END WITH $MODELNAME"
 echo "=================="
 echo "=================="

done

