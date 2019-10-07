from __future__ import print_function, division
import os
import sys
import datetime

fishIdx = [("e", 2),  ("e", 5), ("c", 1),  ("c", 6),  ("enp", 1), ("enp", 5)]
# %%
if len(sys.argv)==1:
    print("""usage: python main.py <fish indicator> <model name>
example: python main.py 0 freeze""")
    indicator = 0
    gpu_idx = str(1)
    exit()
else:
    # gpu_idx = sys.argv[1]
    indicator = int(sys.argv[1])
    model = sys.argv[2]

if model=="skip":
    from babelfish.model.deep_skip import DeepSkip, train
    Model = DeepSkip
elif model=='kSVD':
    from babelfish.model.deep_kSVD import Deep_KSVD, train
    Model = Deep_KSVD
elif model=='freeze':
    from babelfish.model.deep_freeze import DeepFreeze, train, trainBoth
    Model = DeepFreeze
    train = trainBoth


# assert int(gpu_idx) in [0,1,2,3] # GPU to use
# os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

fidx = indicator
#%%
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
from tqdm import tqdm
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
# import apex # https://github.com/NVIDIA/apex.git
# from apex.amp import amp


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
from babelfish.helpers import get_frames_from_z, get_imaging_from_fish, gen_imaging, resize_volume, resize_batch, read_cnmf, no_overlap_idx, train_valid_test_split, train_test_split, pad_imaging

from babelfish.stats import sampleMSE
from babelfish.plot import interpret, plot_model_vs_real, makePredVideo, MSEbyDist

from babelfish.data import ZebraFishData
# from babelfish.deep_kSVD import Deep_KSVD, train
from babelfish.half_precision import network_to_half

fishIdx = [("e", 2),  ("e", 5), ("c", 1),  ("c", 6),  ("enp", 1), ("enp", 5)]

T.backends.cudnn.benchmark = True

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
prev_frames = 5
next_frames = 5
kl_lambda = 5e-4
sparse_lambda=1e-3
lr=1e-3
nepochs = 15
nEmbedding = 20
# batch_size = 6
batch_size = 32

# LOAD DATA
f = all_data[fishIdx[fidx][0]][fishIdx[fidx][1]]

frame_times = T.from_numpy(f.frame_st.mean(1).astype(np.float32))
shocks = T.FloatTensor(frame_times.shape).zero_()
shocks[np.searchsorted(f.frame_et[:,-1], f.shock_st,side="left")] = 1

tail_movements = T.FloatTensor(frame_times.shape).zero_()
tail_movements[np.searchsorted(f.frame_et[:,-1],
    f.tail_movement_start_times,side="left")] = 1

if gen:
    print("Generating fake data...")
    imaging = gen_imaging(32,11,232,512)
elif cnmf:
#     imaging = read_cnmf('/home/ubuntu/f01555')
    imaging = np.load('/home/ubuntu/f01555_cnmf_small.npz')['fish']
else:
    print("Loading {}".format(f.fishid))
    fishpath = '/data2/Data/MPzfish/drn_hb/{}/{}_small.npz'.format(f.fishid, f.fishid)
    try:
        imaging = np.load(fishpath)['fish']
    except:
        imaging = get_imaging_from_fish(f)
        print("Resizing & saving")
        imaging = resize_batch(imaging,0.5,0.5)
        np.savez(fishpath,fish=imaging)

print("imaging shape: {}".format(imaging.shape))
imaging = pad_imaging(imaging, 128, 256)

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


# print("Number of tail movements in test: {}".format(np.array([float(x[1]["tail_movement"]) for x in test_data]).sum()))


# print("len(train_data): {}".format(len(train_data)))

# print("len(test_data): {}".format(len(test_data)))

# batch_size = 40
tensorlib = T
if cuda:
    tensorlib = T.cuda

if half:
    tensor = tensorlib.HalfTensor
else:
    tensor = tensorlib.FloatTensor

conv_model = Model(nZ,H,W,nEmbedding,prev_frames,next_frames, tensor=tensor)
if cuda:
    conv_model.cuda()
if half:
    conv_model = apex.fp16_utils.network_to_half(conv_model)
if multi_gpu:
    conv_model = nn.DataParallel(conv_model)
print("total num params:", np.sum([np.prod(x.shape) for x in conv_model.parameters()]))
# conv_model(data[0][0][None,:,None].cuda()).shape


# WARNING: TEST DATA BEING USED
if model=="kSVD":
    avg_Y_loss, avg_Y_valid_loss = train(conv_model,all_data,test_data,nepochs,lr=lr,
          sparse_lambda=sparse_lambda, half=half, cuda=cuda)
elif model=="skip" or model=="freeze":
    avg_Y_loss, avg_Y_valid_loss = train(conv_model,train_data,test_data,nepochs,lr=lr, kl_lambda=1e-3, half=half, cuda=cuda, batch_size=batch_size, num_workers=num_workers)

now = datetime.datetime.today().strftime('%y%m%d-%I:%M%p')

model_name = "/data2/trained_models/{}_{}_{}_X=t-4:t_Y=t+1,t+5_epochs={}".format(now, f.fishid, model, nepochs) +     "_Y_MSE={:.3E}_Y_val_MSE={:.3E}".format(avg_Y_loss, avg_Y_valid_loss)


T.save(conv_model.state_dict(),model_name+".pt")
print("Saved "+model_name+".pt")

frame = makePredVideo(conv_model,train_data,name=model_name+'_train')
makePredVideo(conv_model,train_data,name=model_name+'_test')
