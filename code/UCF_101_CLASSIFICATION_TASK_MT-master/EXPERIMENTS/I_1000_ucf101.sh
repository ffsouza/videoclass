#!/bin/bash

#$ -cwd
#$ -o output_random_1000_I_relu_0.8_wn_0.3_20_ep.dat
#$ -e error_random_1000_I_relu_0.8_wn_0.3_20_ep.dat
#$ -m a
#$ -m e
#$ -M jjorge@dsic.upv.es
#$ -pe mp 8

# para comprobar la sintaxis -w v
# comprobar recursos => qconf -spl
# info: http://www.uibk.ac.at/zid/systeme/hpc-systeme/common/tutorials/sge-howto.html

time ./runall.sh
