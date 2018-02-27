import glob
listado = glob.glob("/home/jjorge/HOG_FEA/HOG_NPY_500/test_per_class/*/*.npy")
f = open("HOG_NPY_500/test_per_class_npy_file_list.txt", 'wb')
for l in listado:
    f.write(l + "\n")
f.close()


listado = glob.glob("/home/jjorge/HOG_FEA/HOG_NPY_500/train_per_class/*/*.npy")
f = open("HOG_NPY_500/train_per_class_npy_file_list.txt", 'wb')
for l in listado:
    f.write(l + "\n")
f.close()
