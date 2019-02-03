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

import os, sys, datetime
import itertools
# LF_CODE_PATH = os.path.expanduser('~/projects/LFAnalyze/code')
# FT_CODE_PATH = os.path.expanduser('~/projects/fishTrax/code/analysis/')
# FD_CODE_PATH = os.path.expanduser('~/projects/fish_despair_notebooks/src/')
# sys.path.insert(0,LF_CODE_PATH)
# sys.path.insert(0,FT_CODE_PATH)
# sys.path.insert(0,FD_CODE_PATH)

# import passivity_2p_imaging_utils as p2putils


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


def resize_volume(images, fx, fy, interpolation=cv2.INTER_LINEAR):
    im = cv2.resize(images[0], None, fx=fx, fy=fy, interpolation=interpolation)
    new = np.zeros([images.shape[0],im.shape[0],im.shape[1]]).astype(np.float32)
    new[0] = im
    for i, img in enumerate(images[1:]):
        new[i] = cv2.resize(img, None, fx=fx, fy=fy, interpolation=interpolation)
    return new

def resize_batch(images, fx, fy, interpolation=cv2.INTER_LINEAR):
    im = cv2.resize(images[0,0], None, fx=fx, fy=fy, interpolation=interpolation)
    new = np.zeros([images.shape[0],images.shape[1], im.shape[0],im.shape[1]]).astype(np.float32)
    for b, vol in enumerate(images):
        for z, img in enumerate(vol):
            new[b,z] = cv2.resize(img, None, fx=fx, fy=fy, interpolation=interpolation)
    return new

def pad_imaging(imaging, H, W):
    try:
        assert imaging.shape[2] <= H and imaging.shape[3] <= W
    except Exception as e:
        print("H ({}) and W ({}) must be greater than {} and {}".format(H, W, imaging.shape[2], imaging.shape[3]))
        raise e
    new_imaging = np.zeros([imaging.shape[0],imaging.shape[1],H,W])
    if imaging.shape[2]==H:
        pad_top = False
    else:
        pad_top = int(np.floor((H-imaging.shape[2])/2))
        pad_bottom = int(np.ceil((H-imaging.shape[2])/2))
    if imaging.shape[3]==W:
        pad_left = False
    else:
        pad_left = int(np.floor((W-imaging.shape[3])/2))
        pad_right = int(np.ceil((W-imaging.shape[3])/2))

    if not pad_right and not pad_bottom:
        return imaging
    elif pad_right and not pad_bottom:
        new_imaging[:,:,:,pad_left:(-pad_right)] = imaging
    elif pad_bottom and not pad_right:
        new_imaging[:,:,pad_top:(-pad_bottom),:] = imaging
    else:
        new_imaging[:,:,pad_top:(-pad_bottom),pad_left:(-pad_right)] = imaging
    return new_imaging.astype(np.float32)

def pad_image(image, H, W):
    try:
        assert image.shape[0] <= H and image.shape[1] <= W
    except Exception as e:
        print("H ({}) and W ({}) must be less than {} and {}".format(H, W, image.shape[0], image.shape[1]))
        raise e
    new_image = np.zeros([H,W])
    if image.shape[0]==H:
        pad_top = False
    else:
        pad_top = int(np.floor((H-image.shape[0])/2))
        pad_bottom = int(np.ceil((H-image.shape[0])/2))
    if image.shape[0]==W:
        pad_left = False
    else:
        pad_left = int(np.floor((W-image.shape[1])/2))
        pad_right = int(np.ceil((W-image.shape[1])/2))
    if not pad_left and not pad_top:
        return image
    elif pad_right and not pad_bottom:
        new_image[:,pad_left:(-pad_right)] = image
    elif pad_bottom and not pad_right:
        new_image[pad_top:(-pad_bottom),:] = image
    else:
        new_image[pad_top:(-pad_bottom),pad_left:(-pad_right)] = image
    return new_image.astype(np.float32)

def crop_image(image, H, W):
    try:
        assert image.shape[0] >= H and image.shape[1] >= W
    except Exception as e:
        print("H ({}) and W ({}) must be less than {} and {}".format(H, W, image.shape[0], image.shape[1]))
        raise e
    new_image = np.zeros([H,W])
    if image.shape[0]==H:
        crop_top = False
    else:
        crop_top = int(np.ceil((image.shape[0]-H)/2))
        crop_bottom = int(np.floor((image.shape[0]-H)/2))
    if image.shape[0]==W:
        crop_left = False
    else:
        crop_left = int(np.ceil((image.shape[1]-W)/2))
        crop_right = int(np.floor((image.shape[1]-W)/2))
    if not crop_left and not crop_top:
        return image
    elif crop_left and not crop_top:
        new_image = image[:,crop_left:(-crop_right)]
    elif crop_top and not crop_left:
        new_image = image[crop_top:(-crop_bottom),:]
    else:
        new_image = image[crop_top:(-crop_bottom),crop_left:(-crop_right)]
    return new_image.astype(np.float32)


def pad_volume(image, H, W):
    # this should be ND...
    try:
        assert image.shape[1] <= H and image.shape[2] <= W
    except Exception as e:
        print("H ({}) and W ({}) must be less than {} and {}".format(H, W, image.shape[1], image.shape[2]))
        raise e
    new_image = np.zeros([image.shape[0], H,W])
    if image.shape[1]==H:
        pad_top = False
    else:
        pad_top = int(np.floor((H-image.shape[1])/2))
        pad_bottom = int(np.ceil((H-image.shape[1])/2))
    if image.shape[1]==W:
        pad_left = False
    else:
        pad_left = int(np.floor((W-image.shape[2])/2))
        pad_right = int(np.ceil((W-image.shape[2])/2))
    if not pad_right and not pad_bottom:
        return image
    elif pad_right and not pad_bottom:
        new_image[:, :,pad_left:(-pad_right)] = image
    elif pad_bottom and not pad_right:
        new_image[:, pad_top:(-pad_bottom),:] = image
    else:
        new_image[:, pad_top:(-pad_bottom),pad_left:(-pad_right)] = image
    return new_image.astype(np.float32)

def norm01(a):
    return (a-a.min())/a.max()

# def shift_image(a,dW,dH):
#     new_a = np.zeros(*a.shape)


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

def caiman_vec_to_2D(x, H, W):
    return np.transpose(x.reshape(W,H))

def caiman_px_to_dl_px(image, corners):
    """Convert from Caiman image space, to DL image space
    ie 0.5 downsample, crop, pad to 256 x 256
    """
    cropped = image[corners[0,0]:corners[1,0],corners[0,1]:corners[3,1]]
    resized = cv2.resize(cropped, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    padded = pad_image(resized, 256, 256)
    return padded
