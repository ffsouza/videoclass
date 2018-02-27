
import numpy as np
import glob
import sys

def evaluate(file_patern_labels, file_patern_ids, path_to_output_representation, debug=False):

	out_repr = np.load(path_to_output_representation)
	labels_pred = np.argmax(out_repr,1);

	total_test_videos = 3783

	filenames = glob.glob(file_pattern_labels)
	filenames.sort()
	numfiles = len(filenames)

	assert numfiles > 0, 'num files labels = %d' % (numfiles)

	labels_true = np.load(filenames[0])


	acum=labels_true.shape[0]

	if(debug):
		print filenames[0]
		print labels_true.shape

	for filename in filenames[1:]:

		curr = np.load(filename)
		labels_true = np.concatenate((labels_true, curr), axis=0)
		acum+=curr.shape[0]

		if(debug):
			print "FILENAME: " + filename
			print "CURRENT: " + str(curr.shape)
			print "TOTAL: " + str(labels_true.shape)
		
	if(debug):	
		print labels_true.shape
		print acum

	# check if labels start with 1 instead of with 0
	if np.min(labels_true) == 1:
		print "Start with 1"
	 	#labels_pred = np.add(labels_pred,1)

	# load ids

	filenames = glob.glob(file_pattern_ids)
	filenames.sort()
	numfiles = len(filenames)

	assert numfiles > 0, 'num files ids = %d' % (numfiles)


	#labels_true = np.zeros(labels_pred.shape[0])
	ids = np.load(filenames[0])

	acum=ids.shape[0]

	if(debug):	
		print filenames[0]
		print ids.shape

	for filename in filenames[1:]:
		curr = np.load(filename)
		ids = np.concatenate((ids, curr), axis=0)
		acum+=curr.shape[0]
		if(debug):	
			print "FILENAME: " + filename
			print "CURRENT: " + str(curr.shape)
			print "TOTAL: " + str(ids.shape)
	
	if(debug):		
		print ids
		print acum


	unique_ids = np.unique(ids)

	acc_voting = 0

	acc_sum = 0

	for unique_id in unique_ids:

		
		
		labels_true_curr_id = labels_true[np.nonzero(ids==unique_id)[0]]
		labels_pred_curr_id = labels_pred[np.nonzero(ids==unique_id)[0]]

		assert len(np.unique(labels_true_curr_id)) == 1, 'Error with labels = %d' % (len(np.unique(labels_true_curr_id)))

		if(debug):
			print "Video id:  " + str(unique_id)
			print "Class: " + str(labels_true_curr_id[0])
			print len(np.nonzero(ids==unique_id)[0])

		"""
		print np.nonzero(labels_true_curr_id==labels_pred_curr_id)[0].shape
		print np.nonzero(labels_true_curr_id!=labels_pred_curr_id)[0].shape

		print np.nonzero(labels_true_curr_id==labels_pred_curr_id)[0].shape
		print np.nonzero(labels_true_curr_id!=labels_pred_curr_id)[0].shape
		"""

		#With numpy 1.9
		#Voting
		unique, counts = np.unique(labels_pred_curr_id, return_counts=True)
		class_counts = zip(counts,unique)
		class_counts.sort(reverse=True)

		if(debug):
			print class_counts

		if( class_counts[0][1] == labels_true_curr_id[0]):
			acc_voting +=1



		#Sum. Posteriors
		out_rep_curr_id = out_repr[np.nonzero(ids==unique_id)[0]]
		sum_curr_posteriors = np.argsort(np.sum(out_rep_curr_id, axis=0))

		if(debug):
			print sum_curr_posteriors

		if( sum_curr_posteriors[-1] == labels_true_curr_id[0]):
			acc_sum +=1

	if(debug):
		print labels_true.shape
		print np.max(labels_true)
		print np.min(labels_true)

		print labels_pred.shape
		print np.max(labels_pred)
		print np.min(labels_pred)

		print len(unique_ids)

		

	#Results
	print labels_true
	acc = sum(labels_true == labels_pred)/float(labels_true.shape[0])
	print 'Test acc : %.5f %%' % (acc * 100.0)

	acc_voting = acc_voting/float(total_test_videos)
	print '(ID-VOTING) Test acc : %.5f %%' % (acc_voting  * 100.0)

	acc_sum = acc_sum/float(total_test_videos)
	print '(ID-SUM) Test acc : %.5f %%' % (acc_sum  * 100.0)

	print "Total # of features: " + str(labels_true.shape[0])
	print "Total # of videos: " + str(total_test_videos)




if __name__ == "__main__":

	if len(sys.argv) < 3:
	    print 'the input format is: python evaluate.py file_pattern_labels file_pattern_ids path_to_output_representation {debug}'
	    print 'a demo is : python evaluate.py "SOME/PLACE/test_labels_*" "SOME/PLACE/test/test_ids_*" net/512_trainall/test/output_layer-00001-of-00001.npy 0'
	    print '(debug = 0 (no debug) or 1 (debug mode))'

	else:

		
		file_pattern_labels = sys.argv[1]
		file_pattern_ids    = sys.argv[2]
		path_to_output_representation = sys.argv[3]

		if(len(sys.argv) > 4):

			if(sys.argv[4] == 1):
				debug = True
			else:
				debug = False

		else:

			debug = False

		evaluate(file_pattern_labels, file_pattern_ids, path_to_output_representation, debug)

