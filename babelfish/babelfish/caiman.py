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

def caiman_loss(pred, spatial, frame, bg, mask):
    """ Calculate loss for apples-to-apples comparison with deep-skip
    pred: vector of ncomponents
    spatial: npixels x ncomponents ['A']
    frame: (T x) H x W
    """
    pred_frame = spatial @ pred + bg
    pred_frame = caiman_vec_to_2D(pred_frame)
    # make same dimensions as deep-skip training
    mask_dl_px = caiman_px_to_dl_px(mask, corners_cnmf)
    frame_dl_px = caiman_px_to_dl_px(frame, corners_cnmf)
    pred_frame_dl_px = caiman_px_to_dl_px(pred_frame, corners_cnmf)
    return np.sqrt(np.sum((frame_dl_px*mask_dl_px-pred_frame_dl_px*mask_dl_px)**2))
