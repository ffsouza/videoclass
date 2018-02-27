#!/bin/bash

for i in 0 1 2 3 4 5 6 7 8 9;
do
echo $i
qsub ./part_launcher.sh data_info/trainPart$i.txt data_info/testPart$i.txt
done
