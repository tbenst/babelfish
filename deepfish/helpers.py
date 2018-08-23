from __future__ import print_function, division
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

import os, sys, datetime
import itertools
LF_CODE_PATH = os.path.expanduser('~/projects/LFAnalyze/code')
FT_CODE_PATH = os.path.expanduser('~/projects/fishTrax/code/analysis/')
FD_CODE_PATH = os.path.expanduser('~/projects/fish_despair_notebooks/src/')
sys.path.insert(0,LF_CODE_PATH)
sys.path.insert(0,FT_CODE_PATH)
sys.path.insert(0,FD_CODE_PATH)

import passivity_2p_imaging_utils as p2putils


def get_frames_from_z(z, fish,half=False):
    tiff = fish.get_tif_rasl(z)
    ntime = fish.frame_et.shape[0]
    if half:
        dtype = np.float16
    else:
        dtype = np.float32
    frames = np.zeros((ntime, tiff.frame_shape[0],tiff.frame_shape[1])).astype(dtype)
    for t in range(ntime):
        frame = np.array(tiff.get_frame(t)).astype(dtype)
        frames[t] = frame
    return frames

def get_imaging_from_fish(f,n_jobs=8, half=False):
    nZ = f.num_zplanes
    if half:
        dtype = np.float16
    else:
        dtype = np.float32
    # frames_by_z = pool.map(partial(get_frames_from_z, fish=f), range(nZ))
    frames_by_z = Parallel(n_jobs=n_jobs)(delayed(get_frames_from_z)(z,fish=f) for z in range(nZ))
    imaging = np.stack(frames_by_z).swapaxes(0,1).astype(dtype)
    return imaging

def gen_imaging(nT, nZ, H, W, half=False):
    if half:
        dtype = np.float16
    else:
        dtype = np.float32
    return np.random.randint(0,3000,[nT,nZ,H,W]).astype(dtype)


def resize_volume(images, fx, fy, interpolation=cv2.INTER_CUBIC):
    im = cv2.resize(images[0], None, fx=fx, fy=fy, interpolation=interpolation)
    new = np.zeros([images.shape[0],im.shape[0],im.shape[1]]).astype(np.float32)
    new[0] = im
    for i, img in enumerate(images[1:]):
        new[i] = cv2.resize(img, None, fx=fx, fy=fy, interpolation=interpolation)
    return new

def resize_batch(images, fx, fy, interpolation=cv2.INTER_CUBIC):
    im = cv2.resize(images[0,0], None, fx=fx, fy=fy, interpolation=interpolation)
    new = np.zeros([images.shape[0],images.shape[1], im.shape[0],im.shape[1]]).astype(np.float32)
    for b, vol in enumerate(images):
        for z, img in enumerate(vol):
            new[b,z] = cv2.resize(img, None, fx=fx, fy=fy, interpolation=interpolation)
    return new

def read_cnmf(base_filename, nZ=11):
    planes = []
    for i in range(1,nZ+1):
        plane = np.load(base_filename + "_plane{}_denoised.mp4.npy".format(i))
        planes.append(plane)
    return np.stack(planes,1)


def no_overlap_idx(startIdx, stopIdx, prev_frames=5, next_frames=5):
    start = startIdx + prev_frames -1
    stop = stopIdx - next_frames
    return list(np.arange(start,stop))

def train_valid_test_split(nIdx, prev_frames=5, next_frames=5, n_per_sample=10, nchunks=3):
    """Eg 5 prev frames and 5 next frames is n_per_sample=10. No overlap of index.
    uses nchunks for validation + nchunks for test"""
    idx_per_chunk = prev_frames+next_frames+n_per_sample - 1
    idx_per_train_chunk =int(( nIdx - (idx_per_chunk*2*nchunks) )/(2*nchunks+1))
    try:
        assert idx_per_chunk*nchunks*2 + idx_per_train_chunk*(2*nchunks+1) <= nIdx
    except:
        print("Need {} indices".format(idx_per_chunk*nchunks*2 + idx_per_train_chunk*(2*nchunks+1)))
        raise
    tvt = {"train": [], "validation": [], "test": []}
    chunk_start = []
    idx = 0
    for i in range(nchunks*2):
        idx += idx_per_train_chunk
        chunk_start.append(idx)
        idx += idx_per_chunk
    chunk_stop = list(map(lambda start: start+idx_per_chunk, chunk_start))
    prev_stop = 0
    for i, (start, stop) in enumerate(zip(chunk_start, chunk_stop)):
        tvt["train"] += no_overlap_idx(prev_stop, start, prev_frames, next_frames)
        if i%2==0:
            tvt["test"] += no_overlap_idx(start, stop, prev_frames, next_frames)
        else:
            tvt["validation"] += no_overlap_idx(start, stop, prev_frames, next_frames)
        prev_stop = stop
    tvt["train"] += no_overlap_idx(prev_stop, nIdx, prev_frames, next_frames)
    return tvt

def train_test_split(nIdx, prev_frames=5, next_frames=5, n_per_sample=10, nchunks=3):
    """Eg 5 prev frames and 5 next frames is n_per_sample=10. No overlap of index.
    uses nchunks for validation + nchunks for test"""
    idx_per_chunk = prev_frames+next_frames+n_per_sample - 1
    idx_per_train_chunk =int(( nIdx - (idx_per_chunk*nchunks) )/(2*nchunks+1))
    try:
        assert idx_per_chunk*nchunks + idx_per_train_chunk*(nchunks+1) <= nIdx
    except:
        print("Need {} indices".format(idx_per_chunk*nchunks + idx_per_train_chunk*(nchunks+1)))
        raise
    tvt = {"train": [], "validation": [], "test": []}
    chunk_start = []
    idx = 0
    for i in range(nchunks):
        idx += idx_per_train_chunk
        chunk_start.append(idx)
        idx += idx_per_chunk
    chunk_stop = list(map(lambda start: start+idx_per_chunk, chunk_start))
    prev_stop = 0
    for i, (start, stop) in enumerate(zip(chunk_start, chunk_stop)):
        tvt["train"] += no_overlap_idx(prev_stop, start, prev_frames, next_frames)
        tvt["test"] += no_overlap_idx(start, stop, prev_frames, next_frames)
        prev_stop = stop
    tvt["train"] += no_overlap_idx(prev_stop, nIdx, prev_frames, next_frames)
    return tvt
