from __future__ import print_function, division
import os
import sys

fishIdx = [("e", 2),  ("e", 5), ("c", 1),  ("c", 6),  ("enp", 1), ("enp", 5)]

if len(sys.argv)==1:
    indicator = 0
else:
    indicator = int(sys.argv[1]) # 0-3

os.environ['CUDA_VISIBLE_DEVICES'] = str(indicator)
fidx = indicator


import torch as T
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import scipy.linalg
from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import scipy.sparse as sparse
from scipy import stats
import gc
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import seaborn as sb
from pandas import DataFrame
from scipy.spatial import distance
from scipy.cluster import hierarchy
from torchvision.transforms import Resize
import dill
from joblib import Parallel, delayed
import cv2
import resource
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import apex # https://github.com/NVIDIA/apex.git
from apex.amp import amp


import os, sys, datetime
import itertools
LF_CODE_PATH = os.path.expanduser('~/projects/LFAnalyze/code')
FT_CODE_PATH = os.path.expanduser('~/projects/fishTrax/code/analysis/')
FD_CODE_PATH = os.path.expanduser('~/projects/fish_despair_notebooks/src/')
sys.path.insert(0,LF_CODE_PATH)
sys.path.insert(0,FT_CODE_PATH)
sys.path.insert(0,FD_CODE_PATH)

import passivity_2p_imaging_utils as p2putils
reload(p2putils)
tmp_dir = '/tmp/'
all_data = p2putils.get_all_datasets(tmp_dir=tmp_dir)

sys.path.insert(0,".")
from deepfish.helpers import get_frames_from_z, get_imaging_from_fish, gen_imaging, resize_volume, resize_batch, read_cnmf, no_overlap_idx, train_valid_test_split, train_test_split

from deepfish.stats import sampleMSE
from deepfish.plot import interpret, plot_model_vs_real, makePredVideo, MSEbyDist

from deepfish.data import ZebraFishData

fishIdx = [("e", 2),  ("e", 5), ("c", 1),  ("c", 6),  ("enp", 1), ("enp", 5)]

torch.backends.cudnn.benchmark = True

# PARAMETERS
gen = False
# gen = True
cuda=True
# cnmf=True
cnmf=False
half=True
half=False
multi_gpu = True
num_workers = 16
prev_frames = 1
next_frames = 1
kl_lambda = 1e-3
sparse_lambda=5


# LOAD DATA
f = all_data[fishIdx[fidx][0]][fishIdx[fidx][1]]

print("Loading {}".format(f.fishid))

frame_times = T.from_numpy(f.frame_st.mean(1).astype(np.float32))
print("YAY")
shocks = T.FloatTensor(frame_times.shape).zero_()
shocks[numpy.searchsorted(f.frame_et[:,-1], f.shock_st,side="left")] = 1

tail_movements = T.FloatTensor(frame_times.shape).zero_()
tail_movements[numpy.searchsorted(f.frame_et[:,-1],
    f.tail_movement_start_times,side="left")] = 1

if gen:
    imaging = gen_imaging(32,11,232,512)
elif cnmf:
#     imaging = read_cnmf('/home/ubuntu/f01555')
    imaging = np.load('/home/ubuntu/f01555_cnmf_small.npz')['fish']
else:
#     imaging = get_imaging_from_fish(f)
# imaging = resize_batch(imaging,0.5,0.5)
# np.savez('/home/ubuntu/f01555.npz',fish=imaging)
# np.savez('/home/ubuntu/f01555_small.npz',fish=imaging)
#     imaging = np.load('/home/ubuntu/f01555.npz')['fish']
    imaging = np.load('/home/ubuntu/f01555_small.npz')['fish']
#     imaging = np.load('/home/ubuntu/f01555_medfilt_small.npy')


# tvt_split = train_valid_test_split(2826, nchunks=20)
tvt_split = train_test_split(2826, nchunks=20)
total_examples = sum([len(x) for x in tvt_split.values()])
print(["{}: {} ({:.2f}%)".format(k, len(v), 100*len(v)/total_examples) for k,v in tvt_split.items()])


# LOAD TIFF
train_data = ZebraFishData(imaging,shocks,tail_movements,
                        tvt_split['train'], prev_frames,next_frames)

# valid_data = ZebraFishData(imaging,shocks,tail_movements,
#                         tvt_split['validation'], prev_frames,next_frames)

test_data = ZebraFishData(imaging,shocks,tail_movements,
                        tvt_split['test'], prev_frames,next_frames)

all_data = ZebraFishData(imaging,shocks,tail_movements,None,
                        prev_frames,next_frames)

_, nZ, H, W = train_data[0][0]["brain"].shape


print("Number of tail movements in test: {}".format(np.array([float(x[1]["tail_movement"]) for x in test_data]).sum()))


print("len(train_data): {}".format(len(train_data)))

print("len(test_data): {}".format(len(test_data)))


nEmbedding = 20
batch_size = 8
batch_size = 40
tensorlib = T
if cuda:
    tensorlib = T.cuda

if half:
    tensor = tensorlib.HalfTensor
else:
    tensor = tensorlib.FloatTensor

conv_model = Deep_KSVD(nZ,H,W,nEmbedding,prev_frames,tensor=tensor)
if cuda:
    conv_model.cuda()
if half:
    conv_model = apex.fp16_utils.network_to_half(conv_model)
if multi_gpu:
    conv_model = nn.DataParallel(conv_model)
print("total num params:", np.sum([np.prod(x.shape) for x in conv_model.parameters()]))
# conv_model(data[0][0][None,:,None].cuda()).shape


# WARNING: TEST DATA BEING USED
# train(conv_model,all_data,test_data,100,lr=1e-3,
      # sparse_lambda=sparse_lambda, half=half, cuda=cuda)
# 1.91E+02


model_name = "180821_kSVD_X=t-4:t_Y=t+1,t+5_epochs={}".format(36) +     "_Y_MSE={:.3E}".format(28.34)


T.save(conv_model.state_dict(),model_name+".pt")

gc.collect()
T.cuda.empty_cache()


mses = sampleMSE(conv_model, test_data, 16)


idx = np.argsort(mses['MSE(X_t,Y_t+1)'])#[-250:]


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
# plt.title("Distribution of MSE for X_pred (t+1)")
# labels = ['MSE(X_pred,X_t-4)', 'MSE(X_pred,X_t-1)', 'MSE(X_pred,X_t)',
#           'MSE(X_pred,Y_t+1)', 'MSE(X_pred,Y_t+4)', 'MSE(X_pred,Y_t+5)']
# vals = [mses[k][idx] for k in labels]
# plt.hist(np.stack(vals,1), 20)
# plt.legend(["{}={:.4g}".format(k,m.mean()) for k, m in zip(labels, vals)])

plt.subplot(2,2,2)
labels = ['MSE(Y_pred,X_t-4)', 'MSE(Y_pred,X_t-1)', 'MSE(Y_pred,X_t)',
          'MSE(Y_pred,Y_t+1)']#, 'MSE(Y_pred,Y_t+4)', 'MSE(Y_pred,Y_t+5)']
vals = [mses[k][idx] for k in labels]
plt.hist(np.stack(vals,1), 20)
plt.legend(["{}={:.4g}".format(k,m.mean()) for k, m in zip(labels, vals)])
plt.title("Distribution of MSE for Y_pred (t+5)")

plt.subplot(2,2,3)
labels = ['MSE(X_t-1,X_t)', 'MSE(X_t,Y_t+1)',
          'MSE(X_t-1,Y_t+1)']
vals = [mses[k][idx] for k in labels]
plt.hist(np.stack(vals,1), 20)
plt.legend(["{}={:.4g}".format(k,m.mean()) for k, m in zip(labels, vals)])
plt.title("Distribution of MSE for different timesteps")

plt.subplot(2,2,4)
# labels = ['MSE(X_pred,Y_pred)']
# vals = [mses[k][idx] for k in labels]
# plt.hist(np.stack(vals,1), 20)
# plt.legend(["{}={:.4g}".format(k,m.mean()) for k, m in zip(labels, vals)])
# plt.title("Distribution of MSE(X_pred,Y_pred)")


for k,v in mses.items():
    if k[:11]=="MSE(Y_pred,":
        print("{}: {:.4g}".format(k,v.mean()))


len(all_data)


plot_model_vs_real(conv_model.module,all_data)

x, y = all_data[1000]
interpret(conv_model.module,x,y,nEmbedding, scale=0.1)

embeddings = plot_embedding_over_time(conv_model.module,train_data)


frame = makePredVideo(conv_model,all_data)
