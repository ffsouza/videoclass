import sys, os
import numpy as np
import commands
import time
import pickle
import gc
import re

DEBUG = False

MAX_PATCHES=100
MAX_FEATURES=435

def logger(line, log, cr=True):

	if(cr):
		print line + "\n"
		#log+= line + "\n"
	else:
		print line
		#log+= line + "\n"

def get_file_lines(filename):
	f = open(filename, 'r')
	file_lines = [line.strip() for line in f.readlines() if line.strip() != '']
	f.close()
	return file_lines

def printInt(number):

	return str(number)

def create_videos_dicc(videos_list):

	videos_list = get_file_lines(videos_list)
	videos_list = [video.strip().split()[0].split("/")[1] for video in videos_list]
	
	video_dict = dict.fromkeys(videos_list,0)

	video_id = 0

	for video in videos_list:
		video_dict[video] = video_id
		video_id+=1

	return video_dict

def create_class_dict(classInd_lines):

	classInd_lines = get_file_lines(path_to_class_ind_file)
	class_list = [class_name.strip().split()[1] for class_name in classInd_lines]

	class_dict = dict.fromkeys(class_list,0)

	class_id = 0

	for class_name in class_list:
		class_dict[class_name] = class_id
		class_id+=1

	return class_dict

def readFeatures(feature_lines, video_name, videos_dict, samples, labels, ids, label, v_id, log):

	logger("Current video # features: " + printInt(len(feature_lines)), log, False)

	patches = 0
	features = 0

	error = False

	for line in feature_lines:

		feat_line = line.strip().split("\t")[1:]
		if(len(feat_line) != MAX_FEATURES or re.search('[a-zA-Z]', line)):
			logger("[ERROR] Line corrupted with " + printInt(len(feat_line)) + " different features." , log, False)
			
		else:

			samples += [label , v_id] + feat_line
			labels.append(label)
			ids.append(v_id)

			patches += 1

			if(patches == MAX_PATCHES):
				break

	logger("Read # samples: " + printInt(patches), log, False)
	logger("Read class: " + video_name, log, False)
	logger("Read video_id: " + video_name, log, False)
	logger("Count: " + printInt(len(samples)) + ", shaped=" + printInt(len(samples)/(MAX_FEATURES + 2)), log, False)

	del feature_lines

	return samples, labels, ids

def store(samples, labels, ids, output_numpy_dir, video_class, log, test_mode=True):

	directory = output_numpy_dir + "/" + video_class
	if not os.path.exists(directory):
		logger("Creating output dir..." + directory, log)
		os.makedirs(directory)

	#Save numpy matrix and 
	logger("Saving class information...." , log)

	print len(samples)
	# Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
	npsamples = np.matrix(samples, dtype=np.float32).reshape(len(samples)/(MAX_FEATURES + 2), MAX_FEATURES + 2)

	print npsamples[0:5,:]
	np.random.shuffle(npsamples)
	print npsamples[0:5,:]
	np.random.shuffle(npsamples)
	print npsamples[0:5,:]
	np.random.shuffle(npsamples)
	print npsamples[0:5,:]
	# Unsigned integer 16 bits (0 to 65535)
	#npids = np.matrix(ids, dtype=np.uint16).T 
	# Unsigned integer 8 bits (0 to 255)
	#nplabels = np.matrix(labels, dtype=np.uint8).T 

	logger("Samples shape: " + str(npsamples.shape) + ", type=" + str(npsamples.dtype) + ", total size=" + str(npsamples.size) , log)
	#logger("Labels shape: " + str(nplabels.shape) + ", type=" + str(nplabels.dtype) + ", total size=" + str(nplabels.size) , log)
	#logger("Ids shape: " + str(npids.shape) + ", type=" + str(npids.dtype) + ", total size=" + str(npids.size) , log)
	
	label = str(int(labels[0]))


	#if(not test_mode):
	logger("Saving... ", log, False)
	directory + "/data_" + label + ".npy\n"
	np.save(directory + "/data_" + label, npsamples)
	#np.save(directory + "/ids_" + label, npids)
	#np.save(directory + "/labels_" + label, nplabels)

	del npsamples
	#del npids
	#del nplabels

	logger("Saved # samples: " + printInt(len(samples)/(MAX_FEATURES + 2)), log, False)
	logger("Saved class: " + video_class, log, False)
	
	del samples
	del labels
	del ids

	logger("==================================================" , log,)

	return directory + "/data_" + label + ".npy\n"

	




def fea2numpy(videos_list, path_to_train_videos_list, path_to_class_ind_file, features_root, max_patches, output_numpy_dir):


	log=""	

	if not os.path.exists("./log"):
		logger("Creating log dir..." + "./log", log)
		os.makedirs("./log")

	global MAX_PATCHES
	MAX_PATCHES = max_patches

	npy_file_list = ""

	
	#Create videos dictionary
	videos_dict=create_videos_dicc(videos_list)

	total_time_start = time.time()

	#Format: class_name/video_name.avi class_int
	training_list_lines = get_file_lines(path_to_train_videos_list)

	print "Read " + printInt(len(training_list_lines)) + " lines on training list"

	"""
	test_list_lines = get_file_lines(path_to_test_videos_list)

	print "Read " + printInt(len(test_list_lines)) + " lines on test list"
	"""
	#Format: class_int class_name
	classInd_lines = get_file_lines(path_to_class_ind_file)

	print "Read " + printInt(len(classInd_lines)) + " lines on class list"

	class_dict=create_class_dict(path_to_class_ind_file)

	#Starting arrays to store labels, video ids, and samples
	labels = []

	ids = []

	samples = []

	processed_samples = 0

	total_patches = 0

	previus_class = training_list_lines[0].split("/")[0]

	logger("Process starting", log)

	logger("Current group: " + previus_class, log)

	for training_sample in training_list_lines:

		sample_time_start = time.time()

		#example line: ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1
		#['ApplyEyeMakeup', 'v_ApplyEyeMakeup_g01_c01.avi 1']
		trainin_sample_vector_path_video_class = training_sample.split("/")

		#Example:
		#ApplyEyeMakeup
		video_class = trainin_sample_vector_path_video_class[0]
		#v_ApplyEyeMakeup_g01_c01.avi
		video_name =trainin_sample_vector_path_video_class[1].split()[0]
		#1
		video_class_id = class_dict[video_class]
		#0 (v_ApplyEyeMakeup_g01_c01.avi is the first video)
		video_id = videos_dict[video_name]

		if(video_class != previus_class):
			#Other class, store information

			npy_file_list+=store(samples, labels, ids, output_numpy_dir, previus_class, log, True)

			# collected = gc.collect()
			# print "Garbage collector: collected %d objects." % (collected)
			
			labels = []

			ids = []

			samples = []
			
			
			previus_class = video_class

			

			

		logger("Current VIDEO name: " + video_name, log, False)
		logger("Current VIDEO class name: " + video_class, log, False)
		logger("Current VIDEO id: " + printInt(video_id), log, False)
		logger("Current CLASS id: " + printInt(video_class_id), log, False)
		
		

		path_to_file = features_root + "/" + video_class + "/" + video_name.split(".")[0] + ".fea"

		logger("Path to video: " + path_to_file, log, False)


		try:
			#files in trainlistXY that they are not present in features
			feature_file = open(path_to_file, "r")
			feature_lines = feature_file.readlines()
			feature_file.close()

		except IOError as e:
			print "I/O error({0}): {1}".format(e.errno, e.strerror)
			continue
		except:
			print "Unexpected error:", sys.exc_info()[0]
			continue
		

		readFeatures(feature_lines, video_name, videos_dict, samples, labels, ids, video_class_id, video_id, log)

		# patches = 0
		
		# label = video_class_id - 1

		# for line in feature_lines:

		# 	feat_line = line.strip().split("\t")[1:]
		# 	if(len(feat_line) != MAX_FEATURES):
		# 		logger("[ERROR] Line corrupted with " + printInt(len(feat_line)) + " different features." , log, False)
				
		# 	else:

		# 		samples += feat_line
		# 		labels.append(label)
		# 		ids.append(video_id)

		# 		patches += 1

		# 		if(patches == MAX_PATCHES):
		# 			break

		# logger("Read # samples: " + printInt(patches), log, False)
		# logger("Read class: " + video_name, log, False)
		# logger("Read video_id: " + video_name, log, False)
		# logger("Count: " + printInt(len(samples)) + ", shaped=" + printInt(len(samples)/MAX_FEATURES), log, False)

		# del feature_lines

		# sample_time_end = time.time()

		# logger("Time to process the sample: " + str(sample_time_end - sample_time_start) + " segs", log, False)
		
		# logger("", log, False)



	#And the last class
	npy_file_list+=store(samples, labels, ids, output_numpy_dir, video_class, log, True)
		

	#store log
	out = open(output_numpy_dir + "_npy_file_list.txt", 'w')
	out.write(npy_file_list)
	out.close()
	#store log
	out = open("./log/log.txt", 'w')
	out.write(log+'\n')
	out.close()
	



if __name__ == "__main__":

	if len(sys.argv) != 7:
	    print 'the input format is: python fea2numpy.py videos_list path_to_videos_list path_to_class_ind_file features_root max_patches output_numpy_dir'
	    print 'a demo is :python fea2numpy.py data_info/all_videos.txt data_info/trainlist01.txt data_info/classInd.txt ../BACKUP_UCF-101 100 numpy_features'

	else:

		videos_list = sys.argv[1]
		path_to_train_videos_list = sys.argv[2]
		path_to_class_ind_file = sys.argv[3]
		features_root = sys.argv[4]
		max_patches = int(sys.argv[5])
		output_numpy_dir = sys.argv[6]

		fea2numpy(videos_list=videos_list, path_to_train_videos_list=path_to_train_videos_list,\
		 path_to_class_ind_file=path_to_class_ind_file, features_root=features_root, max_patches=max_patches, output_numpy_dir=output_numpy_dir)
