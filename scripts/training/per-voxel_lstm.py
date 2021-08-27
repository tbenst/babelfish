##
# runs well--should be turned into a script?
# main unknown is if 1) data normalization 
from datetime import datetime
from dataclasses import dataclass
import torch
import glia
import tables
import os, sys
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

tif_root = "/scratch/b115"
# tif_dir = None # TODO get from args
# tif_dir = "/scratch/b115/2021-06-29_hsChRmine_6f_6dpf/fish2/TSeries-round3-lrhab-118trial-068"
# tif_dir = "/data/dlab/b115/2021-07-14_rsChRmine_h2b6s_5dpf/fish1/TSeries-lrhab-118trial-061"
# tif_dir = "/data/dlab/b115/2021-07-14_rsChRmine_h2b6s_5dpf/fish1/TSeries-titration-192trial-062"
# tif_dir = f"{tif_root}/2021-07-14_rsChRmine_h2b6s_5dpf/fish2/TSeries-lrhab-118trial-069"
# tif_dir = f"{tif_root}/2021-07-14_rsChRmine_h2b6s_5dpf/fish2/TSeries-titration-192trial-070"
# tif_dir = f"{tif_root}/2021-07-14_rsChRmine_h2b6s_5dpf/fish2/TSeries-cstoner-n64-b2-r8-077"
tif_dir = f"{tif_root}/2021-06-08_rsChRmine_h2b6s/fish2/TSeries-lrhab-118trial-122"

model_base_dir = "/scratch/models/" # TODO: args
gpus = [0] # TODO args
# TODO python fire
batch_size = 4
n_time_steps = 24
max_epochs = 1
# DIVIDE_BY = 512 # good for dim experiments
DIVIDE_BY = 8192 # good for bright
# DIVIDE_BY = 2048 # good compromise..?

MODEL_NAME = f"LSTM_per-voxel-state_divide{DIVIDE_BY}"
MODEL_NAME
##
H, W, Z, T, framePlane2tiffPath = L.tseriesTiffDirMetadata(tif_dir)
green_tiff_paths = sorted(filter(lambda x: "Ch3" in x, glob(tif_dir+"/*.tif")))

##
voltageFiles = L.glob("*VoltageRecording*.csv", tif_dir)
if len(voltageFiles) == 1:
    stim_start_idx, stim_end_idx, frame_start_idx = L.getStimTimesFromVoltages(
        voltageFiles[0], Z)
else:
    stim_start_idx, stim_end_idx, frame_start_idx = L.getStimTimesFromVoltages(
        voltageFiles, Z)
##
frames_to_exclude = np.array([], dtype=np.uint32)
for start, stop in zip(stim_start_idx, stim_end_idx):
    frames_to_exclude = np.concatenate(
        (frames_to_exclude, np.arange(start, stop+1)))
frames_to_exclude
n_exclude_frames = len(frames_to_exclude)
print(
    f"will exclude {n_exclude_frames} frames, or {n_exclude_frames/T:.3f} of total")
##
green_shape = (Z*T, H, W)
green_raw = RawArray(np.ctypeslib.ctypes.c_uint16,
                     int(np.product(green_shape)))

glia.pmap(bf.helpers.read_reshape_and_copy, enumerate(green_tiff_paths),
          initializer=glia.init_worker,
          initargs=(green_raw, green_shape, np.uint16), length=T*Z,
          progress=True)
# green = np.frombuffer(green_raw, dtype=np.uint16).reshape(Z,T,H,W)
green = np.frombuffer(green_raw, dtype=np.uint16).reshape(T, Z, H, W)
# green = np.swapaxes(green, 0,1)
green.shape
##
# plt.imshow(exposure.adjust_gamma(green[stim_start_idx[0]+5, 9], 0.3))


##
# dataset = VolumeSeq(green[:,None,:,16,16], n_time_steps=green.shape[0]-1)
# dataset = VolumeSeq(green[:,None,:,:16,:16], n_time_steps=99)
# dataset = VolumeSeq(green[:,None,:,:16,:16], n_time_steps=green.shape[0]-1)
# add channel singleton
dataset = bf.data.VolumeSeq(green[:, None], frames_to_exclude,
                    n_time_steps=n_time_steps,
                    divide_by=DIVIDE_BY)
# free memory for num_workers > 1...?
del green
n_voxels = np.product(dataset[0][1].shape)
print(f"length: {len(dataset)}")
# dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=batch_size, collate_fn=rnn_collate_fn)
##
# Unfortunately, there's a massive OOM error on CPU/RAM when num_workers>16
# 8 might be optimal...?
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, collate_fn=bf.data.rnn_collate_fn, shuffle=True,
    num_workers=0) # highly variable performance..?

assert os.path.exists(model_base_dir)
now_str = datetime.now().isoformat()
save_dir = os.path.join(model_base_dir,
                        now_str + f"-{MODEL_NAME}")


model = bfm.pixel_lstm.PerVoxel(
        bfm.pixel_lstm.LSTM(n_voxels=n_voxels, batch_size=batch_size), H, W, Z,
                 data_dir=tif_dir, save_dir=save_dir)

# EXPERIMENTAL: init from pretrained....
# => seems to lead to more stim artifact..?
# CKPT_PATH = "/scratch/models/2021-06-21T06:06:13.128963-PerVoxelLSTM_actually_shared-separate_bias_hidden_final.ckpt"
# model = PerVoxel.load_from_checkpoint(CKPT_PATH, data_dir=tif_dir)


neptune_logger = NeptuneLogger(
    api_key=os.environ["NEPTUNE_API_TOKEN"],
    project_name="tbenst/3region-stim",
    params=dict(**{"divide_by": DIVIDE_BY}, **model.hparams),
    experiment_name=MODEL_NAME,  # Optional,
    # tags=["optuna-trial"] + tags
)


checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=save_dir,
    save_top_k=1,
    verbose=False,
    monitor="train loss",
    mode='min',
    prefix='',
    period=1
)

trainer = pl.Trainer(gpus=gpus, gradient_clip_val=0.5,
                     #  truncated_bptt_steps=32,
                     logger=neptune_logger, checkpoint_callback=checkpoint_callback,
                     #  early_stop_callback=PyTorchLightningPruningCallback(
                     #      trial, monitor=monitor),
                     max_epochs=max_epochs, auto_lr_find=True,
                     #  default_root_dir=save_dir
                     )
# trainer.tune(model)
# trainer.fit(model, torch.utils.data.DataLoader(dataset, batch_size=None))
try:
    trainer.fit(model, dataloader)
except Exception as e:
    print(e)
    print("caught exception, saving anyway")
checkpoint_path = save_dir + "_final.ckpt"
trainer.save_checkpoint(checkpoint_path)
sys.stdout.write(checkpoint_path+"\n")
checkpoint_path

##
exit()
##
