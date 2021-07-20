##
from datetime import datetime
from dataclasses import dataclass
import torch
import glia
import tables
import os, sys, re
import pytorch_lightning
import pytorch_lightning.utilities
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning import LightningModule, LightningDataModule
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from glob import glob
from functools import partial
from multiprocessing import RawArray
import matplotlib.pyplot as plt
from skimage import exposure
from julia.api import Julia
from typing import Union, List
jl = Julia(compiled_modules=False)
import pyarrow as pa

# this is fragile and depends on PyCall being installed in root env
from julia import Pkg
Pkg.activate("/home/tyler/code/lensman")
from julia import Lensman

L = Lensman
# very sensivtive to import order
import babelfish as bf
import babelfish_models.models as bfm
import babelfish_models.models.pixel_lstm

# tif_dir = None # TODO get from args
tif_root = "/scratch/b115"
# tif_dir = "/scratch/b115/2021-06-29_hsChRmine_6f_6dpf/fish2/TSeries-round3-lrhab-118trial-068"
tif_dir = f"{tif_root}/2021-07-14_rsChRmine_h2b6s_5dpf/fish1/TSeries-lrhab-118trial-061"
# tif_dir = f"{tif_root}/2021-07-14_rsChRmine_h2b6s_5dpf/fish1/TSeries-titration-192trial-062"
# CKPT_PATH = None  # TODO: read from args
# CKPT_PATH = "/scratch/models/2021-07-10T11:16:38.342868-LSTM_per-voxel-state_divide8192_final.ckpt"
# CKPT_PATH = "/scratch/models/2021-07-18T00:13:19.520229-LSTM_per-voxel-state_divide2048_final.ckpt"
CKPT_PATH = "/scratch/models/2021-07-18T14:40:32.653550-LSTM_per-voxel-state_divide2048_final.ckpt"

model_base_dir = "/scratch/models/"  # TODO: args
gpus = [0]  # TODO args
# TODO python fire
batch_size = 4
n_time_steps = 24
max_epochs = 1
# DIVIDE_BY = 512 # good for dim experiments
# DIVIDE_BY = 8192
DIVIDE_BY = int(re.match(r".*divide(\d*)", CKPT_PATH)[1])
MODEL_NAME = f"LSTM_per-voxel-state_divide{DIVIDE_BY}"

##
H, W, Z, T, framePlane2tiffPath = L.tseriesTiffDirMetadata(tif_dir)
green_tiff_paths = sorted(filter(lambda x: "Ch3" in x, glob(tif_dir+"/*.tif")))
##
green_shape = (Z*T, H, W)
green_raw = RawArray(np.ctypeslib.ctypes.c_uint16,
                     int(np.product(green_shape)))

glia.pmap(bf.helpers.read_reshape_and_copy, enumerate(green_tiff_paths),
          initializer=glia.init_worker,
          initargs=(green_raw, green_shape, np.uint16), length=T*Z,
          progress=True)
green = np.ctypeslib.as_array(green_raw).reshape(T, Z, H, W)
# green = np.frombuffer(green_raw, dtype=np.uint16).reshape(T, Z, H, W)

##
##
best_model = bfm.pixel_lstm.PerVoxel.load_from_checkpoint(CKPT_PATH)
best_model = best_model.cuda()
model = best_model
##
full_dataset = bf.data.VolumeSeq(green[:, None], [],
                                n_time_steps=n_time_steps,
                                divide_by=DIVIDE_BY)
##
with torch.no_grad():
    x, y = full_dataset[100-n_time_steps]
    x = torch.from_numpy(x[:, None]).repeat(1, 4, 1, 1, 1, 1)
    x = x.cuda()
    clean = best_model(x)
    clean = clean[:, 0].cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
im0 = axes[0].imshow(y[0, 5])
im = axes[1].imshow(clean[0, 5])
# fig.colorbar(im0)
fig.colorbar(im)
##
inference_batch_size = 4  # even 8 OOM...?
dataloader = torch.utils.data.DataLoader(
    full_dataset, batch_size=inference_batch_size, collate_fn=bf.data.rnn_collate_fn, shuffle=False,
    num_workers=0, pin_memory=True)

tyh5_path = tif_dir + ".ty.h5"
# tyh5_path = tif_dir + "2021-06-21_6pm.ty.h5"
# tyh5_path = '/data/dlab/b115/2021-06-08_rsChRmine_h2b6s/fish2/TSeries-lrhab-titration-123' + \
#     "2021-06-21_6pm.ty.h5"
# tyh5_path = '/data/dlab/b115/2021-06-08_rsChRmine_h2b6s/fish2/TSeries-lrhab-titration-123' + \
#     "2021-06-21_6pm.ty.h5"
# tyh5_path = "/scratch/b115/2021-06-08_rsChRmine_h2b6s/fish2/TSeries-lrhab-titration-1232021-06-21_6pm.ty.h5"
# TODO: read arg
out_dataset = f"/imaging/{MODEL_NAME}-2021-07-02"
##
# TODO: aren't we missing the first ~24 indices...? => left as 0 I think?
# would it be better to copy original data or ...?
with tables.open_file(tyh5_path, 'a') as tyh5:
    if out_dataset in tyh5:
        # raise ValueError(f"dset ({out_dataset}) already exists")
        print(f"dset ({out_dataset}) already exists. Removing...")
        tyh5.remove_node(out_dataset)
    dset_filter = tables.filters.Filters(complevel=0, complib='blosc:zstd')
    group, dset_name = os.path.split(out_dataset)
    parent, name = os.path.split(group)
    if not group in tyh5:
        h5_group = tyh5.create_group(parent, name, createparents=True)
    else:
        h5_group = tyh5.root[group]

    dtype = np.dtype("float32")
    dset = tyh5.create_carray(group, dset_name,
                              tables.Atom.from_dtype(dtype), shape=green[:, None].shape,
                              )
    #   filters=dset_filter)
    with torch.no_grad():
        for b, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
            x, y = batch_data
            to_pad = batch_size - y.shape[0]
            if to_pad != 0:
                # need to pad. Not yet tested....
                y = F.pad(y, (0, 0, 0, 0, 0, 0, 0, 0, 0, to_pad))
                x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 0, 0, to_pad, 0, 0))
            x = x.cuda()
            clean = best_model(x)
            # clean = tensor_to_uint16(clean)
            # remove channel
            # clean = clean[:, 0].cpu().numpy()
            clean = clean.cpu().numpy()
            s = n_time_steps + b*inference_batch_size
            # TODO: perhaps this makes sense so model is fed info from
            # 1:t, and prediction for t+1 is used as `denoised t`.
            # s -= 1
            dset[s:s+inference_batch_size] = clean

##