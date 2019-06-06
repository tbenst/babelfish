from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div

import cv2
from glob import glob
import re
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import scipy
from skimage.external.tifffile import TiffFile
import sys
import time
import logging

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.DEBUG)

import caiman as cm
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
from caiman.utils.utils import download_demo

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in [9]])
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'


in_folder = "/mnt/fs3/tyler/20181127/ome_test/"
# out_folder = "/home/tyler/Dropbox/data/20181127/f10542"


files = np.array(list(filter(lambda x: x[-3:]=="tif", glob(in_folder+"/*"))))
b

fns = list(map(lambda x: os.path.basename(x), files))
fishname = re.compile("(.*)_.*").match(fns[0]).group(1)
zs = np.array(list(map(lambda x: z_re.match(x).group(1), fns)),dtype=int)

# fnames = list(files[np.argsort(zs)])
fnames = [files[0]]
m_orig = cm.load_movie_chain(fnames)[0:-1]

max_shifts = (6, 6)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
strides =  (48, 48)  # create a new patch every x pixels for pw-rigid correction
overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
num_frames_split = 100  # length in frames of each chunk of the movie (to be processed in parallel)
max_deviation_rigid = 3   # maximum deviation allowed for patch with respect to rigid shifts
pw_rigid = False  # flag for performing rigid or piecewise rigid motion correction
shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)

#%% start the cluster (if a cluster already exists terminate it)
if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

    # create a motion correction object
mc = MotionCorrect(fnames, dview=dview, max_shifts=max_shifts,
                  strides=strides, overlaps=overlaps,
                  max_deviation_rigid=max_deviation_rigid,
                  shifts_opencv=shifts_opencv, nonneg_movie=True,
                  border_nan=border_nan, use_cuda=True)

#%% motion correct piecewise rigid
mc.pw_rigid = True  # turn the flag to True for pw-rigid motion correction

mc.motion_correct(save_movie=True)
m_els = cm.load(mc.fname_tot_els)
