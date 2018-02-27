'''
Created on Nov 28, 2014

@author: zhengyin
'''
import sys, os
import numpy as np
import commands

sys.argv.pop(0)
print 'You need to put THIS python script into the folder that contains DenseTracStab'

if len(sys.argv) != 10:
    print 'the input format is: python sample_features.py path_video n_selected metric(dist or HOF_zerobin) alpha trajectory_temporallength stride neighborhood_size spatial_cells temporal_cells output_path'
    print 'a demo is :python sample_features.py ./video.avi 100 HOF_zerobin 2 15 5 32 2 3 ./output.features'
    exit(0)
video_path = sys.argv[0]
n_samples = int(sys.argv[1])
metric = sys.argv[2]
alpha = float(sys.argv[3])
traj_temporallen = int(sys.argv[4])
step_size = int(sys.argv[5])
neigborhood_size = int(sys.argv[6])
spatial_cells = int(sys.argv[7])
temporal_cells = int(sys.argv[8])
output_file = sys.argv[9]

cmd = './DenseTrackStab %s -L %d -W %d -N %d -s %d -t %d'%(video_path, traj_temporallen, step_size, neigborhood_size, spatial_cells, temporal_cells)
result = commands.getoutput(cmd)
result_split = result.split('\n')
n_features = len(result_split)
# assert(n_features >= n_samples)
if n_features < n_samples:
    print 'Warrning: number you required exceeds the total feature numbers from the video'
    n_samples = n_features
    
index = np.arange(n_features)
HOF_start_ID = 10+2*traj_temporallen+8*spatial_cells*spatial_cells*temporal_cells
HOF_len = 9*spatial_cells*spatial_cells*temporal_cells
#TODO: use inverse zero bin as the metric
if metric == 'HOF_zerobin':
    HOF_zerobinsum_list = []
    for line in result_split:
        tokens = np.array(line.rstrip().split('\t'), np.float32)
        
        HOF = tokens[HOF_start_ID:HOF_start_ID+HOF_len]
        HOF_zerobin_sum = HOF[8::9].sum()
        HOF_zerobinsum_list.append(HOF_zerobin_sum)
    rawmetric = -np.array(HOF_zerobinsum_list, np.float32)
elif metric == 'dist':
    dist_list = []
    for line in result_split:
        tokens = np.array(line.rstrip().split('\t'), np.float32)
        dist_list.append(tokens[5])
    rawmetric = np.array(dist_list, np.float32)
else:
    print 'metric type error'
    exit(-1)
        
rawmetric = rawmetric - rawmetric.max()
a = alpha*rawmetric
f = np.exp(a)
prob = f/f.sum()
prob = prob.astype(np.float64)
prob /= prob.sum()

sel_index = np.random.choice(n_features, n_samples, False, prob)
 
# perm = np.random.permutation(index)
# sel_index = perm[:n_samples]
out = open(output_file, 'w')
for id in sel_index:
    line = result_split[id]
    out.write(line+'\n')
out.close()
print '%d features extracted, %d of them are selected, selecting metric is %s'%(n_features, n_samples, metric)
print 'you can find the selected points here: %s'%(output_file)




# os.system(cmd)

# print 123

