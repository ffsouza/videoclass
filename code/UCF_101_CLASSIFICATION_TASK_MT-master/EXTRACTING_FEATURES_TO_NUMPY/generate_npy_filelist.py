import glob
listado = glob.glob("./HOG_NPY/test_per_class/*/*.npy")
f = open("./HOG_NPY/test_per_class_npy_file_list.txt", 'wb')
for l in listado:
    f.write(l + "\n")
f.close()

'''
listado = glob.glob("./HOG_NPY/train_per_class/*/*.npy")
f = open("./HOG_NPY/train_per_class_npy_file_list.txt", 'wb')
for l in listado:
    f.write(l + "\n")
f.close()
'''
