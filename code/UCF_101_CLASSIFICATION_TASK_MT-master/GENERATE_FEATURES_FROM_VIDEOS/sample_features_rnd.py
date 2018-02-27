'''
Created on Nov 28, 2014

@author: zhengyin
'''
import sys, os
import numpy as np
import commands

sys.argv.pop(0)
print 'You need to put THIS python script into the folder that contains DenseTracStab'

if len(sys.argv) != 9:
    print 'the input format is: python sample_features.py path_video n_selected alpha trajectory_length stride neighborhood_size spatial_cells temporal_cells output_path'
    print 'a demo is :python ./video.avi 100 2 15 5 32 2 3 ./output.features'
video_path = sys.argv[0]
n_samples = int(sys.argv[1])
alpha = float(sys.argv[2])
traj_len = int(sys.argv[3])
step_size = int(sys.argv[4])
neigborhood_size = int(sys.argv[5])
spatial_cells = int(sys.argv[6])
temporal_cells = int(sys.argv[7])
output_file = sys.argv[8]

cmd = './DenseTrackStab %s -L %d -W %d -N %d -s %d -t %d'%(video_path, traj_len, step_size, neigborhood_size, spatial_cells, temporal_cells)
result = commands.getoutput(cmd)
result_split = result.split('\n')
n_features = len(result_split)
assert(n_features >= n_samples)
index = np.arange(n_features)
perm = np.random.permutation(index)
sel_index = perm[:n_samples]
out = open(output_file, 'w')
for id in sel_index:
    line = result_split[id]
    out.write(line+'\n')
out.close()
print '%d features extracted, %d of them are selected'%(n_features, n_samples)
print 'you can find the selected points here: %s'%(output_file)




# os.system(cmd)

print 123

