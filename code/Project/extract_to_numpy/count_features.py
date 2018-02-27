import numpy as np
import glob


listado = glob.glob("HOG_NPY_500/train_per_class/*/*.npy")

counter = 0
for l in listado:
	tr=np.load(l)
	counter+=tr.shape[0]
	del tr

print "Total training features: " + str(counter)


del listado

listado = glob.glob("HOG_NPY_500/test_per_class/*/*.npy")

counter = 0
for l in listado:
	tr=np.load(l)
	counter+=tr.shape[0]
	del tr

print "Total test features: " + str(counter)

