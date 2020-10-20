from sklearn.metrics import mean_squared_error
import numpy as np 
import math 
import random 

def load_depth(gt_path, es_path):
    gt = np.load(gt_path)['arr_0']
    es = np.load(es_path)['arr_0']
    return gt,es

def compare_depth(single_gt,single_es):
    max_gt = np.amax(single_gt)
    min_gt = np.min(single_gt)
    max_es = np.max(single_es)
    min_es = np.min(single_es)
    mse = mean_squared_error(single_gt, single_es)
    mse_log = mean_squared_error(np.log(single_gt), np.log(single_es))
    return max_gt,min_gt,max_es,min_es,mse,mse_log

def compare_multiple_depth(gt,es):
    return 


gt_path = '/Users/ouchouyang/Downloads/NYUv2/test_depth_estimation_densedepth.npz'
es_path = '/Users/ouchouyang/Downloads/NYUv2/test_depth.npz'

gt, es = load_depth(gt_path,es_path)

#train_depth_estimation = np.load('/Users/ouchouyang/Downloads/NYUv2/train_depth_estimation_megadepth.npz')
#train_depth_estimation = train_depth_estimation['arr_0']

#train_depth_ground_truth = np.load('/Users/ouchouyang/Downloads/NYUv2/train_depth.npz')
#train_depth_ground_truth = train_depth_ground_truth['arr_0']

mse = 0

random_index = random.randint(0,gt.shape[0]-1) 
#for i in range(train_depth_estimation.shape[0]):
    
A = gt[random_index,:,:]
B = es[random_index,:,:]

print(np.amax(A))
print(np.amax(B))
print(np.amin(A))
print(np.amin(B))

#break
for i in range(gt.shape[0]):

    mse += mean_squared_error(gt[i], es[i])

mse = mse/ (gt.shape[0])
print(mse)