import numpy as np
import os
import glob
from scipy import stats
import math
import sys

def normalize(A, u=None, st=None):

	if(u==None and st==None):
		u=np.mean(A,0)
		st=np.std(A,0, ddof=1)

	Am=A - u[np.newaxis,:]

	return Am/st,u,st


# for l in lineas_tr[1:]:
#     tr=np.concatenate((tr, np.load(l)),axis=0)
#     print tr.shape

def process_data(path_to_file_list, output_path, prefix_filename, u=None, st=None, logger=None, testmode=False):




	# 
	# f_te = open("numpy_features/test/testnpy_file_list.txt",'r')
	# lineas_tr = [l.strip() for l in f_tr.readlines()]
	# lineas_te = [l.strip() for l in f_te.readlines()]
	# f_tr.close()
	# f_te.close()

	print path_to_file_list
	print output_path
	
	f_tr = open(path_to_file_list,'r')
	lineas_tr = [l.strip() for l in f_tr.readlines()]
	f_tr.close()

	acum=0

	tr=np.load(lineas_tr[0])

	print tr.shape

	for l in lineas_tr[1:]:
	    tr=np.concatenate((tr, np.load(l)),axis=0)
	    print tr.shape

	original_shape = tr.shape
	print np.max(tr[:,0])
	print np.min(tr[:,0])
	np.random.shuffle(tr)
	print np.max(tr[:,0])
	print np.min(tr[:,0])
	np.random.shuffle(tr)
	print np.max(tr[:,0])
	print np.min(tr[:,0])
	np.random.shuffle(tr)
	print np.max(tr[:,0])
	print np.min(tr[:,0])

	label_mat=tr[:,0].astype(np.uint8)
	ids_mat=tr[:,1].astype(np.uint16)
	data_mat=tr[:,2:]

	del tr

	#Normalize
	#data_mat_norm=stats.zscore(data_mat, axis=0, ddof=1)
	if(u==None and st == None):
		precomputed_u_and_st=False
		data_mat_norm,u,st =normalize(data_mat)
	else:
		precomputed_u_and_st=True
		data_mat_norm,_,_ =normalize(data_mat, u, st)

	data_mat_max= np.max(data_mat)
	data_mat_min= np.min(data_mat)	
	data_mat_shape= data_mat.shape

	print
	print np.max(label_mat)
	print np.max(ids_mat)
	print np.max(data_mat)

	print
	print np.min(label_mat)
	print np.min(ids_mat)
	print np.min(data_mat)

	print
	print np.max(data_mat_norm)
	print np.min(data_mat_norm)

	print len(np.mean(data_mat_norm,0))
	print len(np.std(data_mat_norm,0))
	print data_mat.shape

	del data_mat

	split_size=1000

	n_d=len(str(data_mat_shape[0]/split_size))

	indexes=[format(a, "0"+str(n_d) + "d") for a in range(0, (data_mat_shape[0]/split_size) + 1)]

	directory=output_path
	print directory
	if not os.path.exists(directory):
		print "Creating output directory..." + directory
		os.makedirs(directory)


	n_packets = data_mat_norm.shape[0]/split_size
	remain = data_mat_norm.shape[0]%split_size

	print "N_packets: " + str(n_packets)
	print "Remain: " + str(remain)

	tot_features_tr =0
	tot_files_tr=0

	for k in range(n_packets):
		#00->99
		lower=k * split_size
		upper=lower + split_size

		print "L: " + str(lower)
		print "U: " + str(upper)
		print "Q: " + str(upper-lower)

		labels_slim_mat=label_mat[lower:upper]	
		ids_slim_mat=ids_mat[lower:upper]	
		data_slim_mat=data_mat_norm[lower:upper,:]

		print labels_slim_mat.shape
		print ids_slim_mat.shape
		print data_slim_mat.shape


		if(not testmode):
			np.save(directory + "/" + prefix_filename + "_data_" + indexes[k], data_slim_mat)
			np.save(directory + "/" + prefix_filename + "_labels_" + indexes[k], labels_slim_mat)
			np.save(directory + "/" + prefix_filename + "_ids_" + indexes[k], ids_slim_mat)

		print "Stored on: " + directory + "/" + prefix_filename + "_ids_"         + indexes[k]
		print "Stored on: " + directory + "/" + prefix_filename + "_labels_"       + indexes[k]
		print "Stored on: " + directory + "/" + prefix_filename + "_data_"         + indexes[k]

		tot_features_tr += data_slim_mat.shape[0]
		tot_files_tr+=1
		print "Previus k: " + str(k)

		
	if(remain != 0):
		print "Inside k: " + str(k)
		k+=1
		labels_slim_mat=label_mat[label_mat.shape[0] - remain:]	
		ids_slim_mat=ids_mat[ids_mat.shape[0] - remain:]		
		data_slim_mat=data_mat_norm[data_mat_norm.shape[0] - remain:,:]	

		print "Stored on: " + directory + "/" + prefix_filename + "_ids_"         	+ indexes[k]
		print "Stored on: " + directory + "/" + prefix_filename + "_labels_"       + indexes[k]
		print "Stored on: " + directory + "/" + prefix_filename + "_data_"         + indexes[k]

		print labels_slim_mat.shape
		print ids_slim_mat.shape
		print data_slim_mat.shape

		tot_features_tr += data_slim_mat.shape[0]

		if(not testmode):
			np.save(directory + "/" + prefix_filename + "_data_" + indexes[k], data_slim_mat)
			np.save(directory + "/" + prefix_filename + "_labels_" + indexes[k], labels_slim_mat)
			np.save(directory + "/" + prefix_filename + "_ids_" + indexes[k], ids_slim_mat)

		tot_files_tr+=1

	print
	print np.max(label_mat)
	print np.max(ids_mat)
	print data_mat_max

	print
	print np.min(label_mat)
	print np.min(ids_mat)
	print data_mat_min

	print
	print np.max(data_mat_norm)
	print np.min(data_mat_norm)

	print len(np.mean(data_mat_norm,0))
	print len(np.std(data_mat_norm,0))
	print data_mat_shape

	del label_mat
	del ids_mat
	del data_mat_norm

	print "TOTAL FEATURES: " + str(tot_features_tr)
	logger.write("TOTAL FEATURES: " + str(tot_features_tr) + "\n")

	print "TOTAL FILES: " + str(tot_files_tr)
	logger.write("TOTAL FILES: " + str(tot_files_tr) + "\n")

	lista=glob.glob(directory + "/" + prefix_filename + "_data_*")

	logger.write("CHECK FILES: " + str(len(lista)) + "\n")
	logger.write(str(lista))

	print "(ON ORIGINAL MAT): " + str(original_shape)
	logger.write("(ON ORIGINAL MAT): " + str(original_shape) + "\n")

	if(not precomputed_u_and_st):
		print "(Normalized with computed u and std.dev)"
	else:
		print "(Normalized with given u and std.dev)"

	return u,st


def launch(path_to_file_train_list, prefix_train_filename, path_to_file_test_list, prefix_test_filename, output_path, testmode=True):

	logger=open(output_path + '/stats.txt','w')

	u,st = process_data(path_to_file_train_list, output_path + "/train", prefix_train_filename, None, None, logger, testmode)

	process_data(path_to_file_test_list, output_path + "/test", prefix_test_filename, u, st, logger, testmode)

	logger.close() 







if __name__ == "__main__":

	if len(sys.argv) < 6:
	    print 'the input format is: python prepare_features.py path_to_file_train_list prefix_train_filename path_to_file_test_list prefix_test_filename output_path {testmode}'
	    print 'a demo is :python prepare_features.py numpy_features_hog/trainnpy_file_list.txt trainall numpy_features_hog/testnpy_file_list.txt test HOG_features/features_data 0'
	    print '{testmode} = 0 (store on disk) or 1 (do not store on disk)'
	else:

		"""
		path_to_file_train_list="numpy_features_hog/trainnpy_file_list.txt"
		prefix_train_filename="trainall"
		path_to_file_test_list="numpy_features_hog/testnpy_file_list.txt"
		prefix_test_filename="test"
		output_path="../data"
		testmode=False
		"""
		path_to_file_train_list=sys.argv[1]
		prefix_train_filename=sys.argv[2]
		path_to_file_test_list=sys.argv[3]
		prefix_test_filename=sys.argv[4]
		#output_path=sys.argv[5]
		output_path="../data"

		if(len(sys.argv) > 6):

			if(sys.argv[6]=="1"):
				testmode=True
			else:
				testmode=False
		else:
			testmode=False

		launch(path_to_file_train_list, prefix_train_filename, path_to_file_test_list, prefix_test_filename, output_path, testmode)
		
