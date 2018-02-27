for i in $(cat $1); do 
    for j in $(ls ~/UCF-101/$i/*.avi); do
	echo $j; 
	if [ -e ${j%%.avi}.fea ]; then 
	    echo ${j%%.avi}.fea "exits";
	else
	    python sample_features.py $j 100 2 15 5 32 2 3 ${j%%.avi}.fea
	fi
    done
done

