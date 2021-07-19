##
# error("TODO: save Uint16 for space savings / avoid major headache")
error "don't use--old/legacy"
from datetime import datetime
from dataclasses import dataclass
import torch
import glia
import tables
import os
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

# this is fragile and depends on PyCall being installed in root env
from julia import Pkg
Pkg.activate("/home/tyler/code/lensman")
from julia import Lensman

L = Lensman
##
# tif_dir = "/data/dlab/b115/2020-10-28_elavl3-chrmine-Kv2.1_h2b6s_8dpf/fish1/TSeries-lrhab_raphe_stim-40trial-038"
# tif_dir = "/scratch/b115/2021-05-18_rsChRmine_h2b6s_6dpf/fish5/TSeries-lrhab-control-91trial-4Mhz-045"
# tif_dir = "/data/dlab/b115/2021-06-02_rsChRmine-h2b6s/fish2/TSeries-titration-192trial-062"
# tif_dir = "/data/dlab/b115/2021-06-02_rsChRmine-h2b6s/fish2/TSeries-titration-192trial-062"
# tif_dir = "/scratch/b115/2021-06-08_rsChRmine_h2b6s/fish2/TSeries-lrhab-titration-123"

model_base_dir = "/scratch/models/"
gpus = [0]
# best performance (i think? at least approx most that fits in RAM...)
batch_size = 4
n_time_steps = 24
max_epochs = 1
# faster
# batch_size = 8
# n_time_steps = 12
# max_epochs = 1

##


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


##
H, W, Z, T, framePlane2tiffPath = L.tseriesTiffDirMetadata(tif_dir)
green_tiff_paths = sorted(filter(lambda x: "Ch3" in x, glob(tif_dir+"/*.tif")))
##


def read_and_copy(index_path_tuple, k=lambda x: x):
    index, path = index_path_tuple
    glia.config.worker_args[0][index] = k(plt.imread(path))


def read_reshape_and_copy(index_path_tuple, k=lambda x: x):
    index, path = index_path_tuple
    X = np.frombuffer(glia.config.worker_args[0],
                      dtype=glia.config.worker_args[2]).reshape(glia.config.worker_args[1])
    X[index] = k(plt.imread(path))


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

glia.pmap(read_reshape_and_copy, enumerate(green_tiff_paths),
          initializer=glia.init_worker,
          initargs=(green_raw, green_shape, np.uint16), length=T*Z,
          progress=True)
# green = np.frombuffer(green_raw, dtype=np.uint16).reshape(Z,T,H,W)
green = np.frombuffer(green_raw, dtype=np.uint16).reshape(T, Z, H, W)
# green = np.swapaxes(green, 0,1)
green.shape
##
plt.imshow(exposure.adjust_gamma(green[stim_start_idx[0], 9], 0.3))
##


def rnn_collate_fn(tensors):
    "Use for LSTM where need to "
    # print(f"{len(tensors)}")
    # print(f"{len(tensors[0])}")
    # print(f"{len(tensors[0][0])}")
    # seq x batch x feature (... x feature)
    Xs = torch.stack([torch.from_numpy(t[0]) for t in tensors], dim=1)
    # batch x feature (... x feature)
    Ys = torch.stack([torch.from_numpy(t[1]) for t in tensors], dim=0)
    return (Xs, Ys)


def make_index_map(n, exclude_idxs):
    "Create a map of length n - exclude_idxs."
    new_idx = 0
    idx_map = {}
    exclude_idxs = sorted(exclude_idxs)
    for i in range(n):
        if len(exclude_idxs) > 0 and i == exclude_idxs[0]:
            exclude_idxs.pop(0)
        else:
            idx_map[new_idx] = i
            new_idx += 1
    return idx_map


assert make_index_map(5, [2, 3]) == {0: 0, 1: 1, 2: 4}


class VolumeSeq(Dataset):
    # volumes:np.ndarray # T x C x Z x H x W

    def __init__(self, volumes: np.ndarray, exclude_frames_from_y: List,
                 n_time_steps: int = 10):
        self.n_time_steps = n_time_steps
        self.volumes = volumes

        n_exclude = len(exclude_frames_from_y)
        self.length = self.volumes.shape[0] - self.n_time_steps - n_exclude
        self.idx_map = make_index_map(self.length, exclude_frames_from_y)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        last_idx = i+self.n_time_steps
        # we add batch singleton dimension
        x = self.volumes[i:last_idx].astype(np.float32)
        x /= 8192
        y = self.volumes[last_idx].astype(np.float32)
        y /= 8192
        return (x, y)


# dataset = VolumeSeq(green[:,None,:,16,16], n_time_steps=green.shape[0]-1)
# dataset = VolumeSeq(green[:,None,:,:16,:16], n_time_steps=99)
# dataset = VolumeSeq(green[:,None,:,:16,:16], n_time_steps=green.shape[0]-1)
# add channel singleton
dataset = VolumeSeq(green[:, None], frames_to_exclude, n_time_steps=n_time_steps)
n_voxels = np.product(dataset[0][1].shape)
print(f"length: {len(dataset)}")
# dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=batch_size, collate_fn=rnn_collate_fn)
##


class PerVoxel(pl.LightningModule):
    """Make a neural net operate on each voxel in a volume."""

    def __init__(self, net, H, W, Z, lr=5e-4, n_loss_timesteps=8,
                 data_dir=""):
        super().__init__()
        self.H = H
        self.W = W
        self.Z = Z
        self.lr = lr
        self.net = net
        # must be sma
        self.n_loss_timesteps = n_loss_timesteps
        self.data_dir = data_dir
        self.save_hyperparameters()

    def forward(self, x):
        # print(f"{x.shape}")
        T, B, C, Z, H, W = x.shape
        # assert T < self.n_loss_timesteps
        x = torch.moveaxis(x, 2, -1)  # channel last
        x = x.reshape(T, B*H*W*Z, C)
        out = self.net(x)[-1]  # only last output
        out = out.reshape(B, Z, H, W, C)
        return torch.moveaxis(out, -1, 1)

    def loss(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        recon_loss = F.mse_loss(
            out, y, reduction='sum') / x.shape[0]
        # recon_loss = recon_loss.detach().item() # does this reduce memory?
        return {"loss": recon_loss}

    def training_step(self, batch, batch_idx):
        loss_dict = self.loss(batch, batch_idx)
        loss = loss_dict["loss"]
        self.log("train loss", loss, prog_bar=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     loss_dict = self.loss(batch, batch_idx)
    #     loss = loss_dict["loss"]
    #     result = pl.EvalResult(checkpoint_on=loss)
    #     result.log('loss', loss, on_epoch=True)
    #     return result

    # def test_step(self, batch, batch_idx):
    #     loss_dict = self.calc_loss(batch, batch_idx)
    #     return loss_dict["loss"]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=1, output_size=1,
                 batch_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        # TODO allow per voxel parameter & only broadcast on batch dimension
        self.h0 = nn.Parameter(torch.zeros(1, 1, self.hidden_layer_size))
        self.c0 = nn.Parameter(torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        bz = input_seq.shape[1]
        lstm_out, _ = self.lstm(
            input_seq, (self.h0.repeat(1, bz, 1), self.c0.repeat(1, bz, 1)))
        return lstm_out


# TODO: exclude indices with stim artifact from prediction/y
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, collate_fn=rnn_collate_fn, shuffle=True,
    num_workers=1, pin_memory=True)
model = PerVoxel(LSTM(batch_size=n_voxels*batch_size), H, W, Z,
                 data_dir=tif_dir)
MODEL_NAME = "PerVoxelLSTM"
##
neptune_logger = NeptuneLogger(
    api_key=os.environ["NEPTUNE_API_TOKEN"],
    project_name="tbenst/3region-stim",
    params=model.hparams,
    experiment_name=MODEL_NAME,  # Optional,
    # tags=["optuna-trial"] + tags
)

assert os.path.exists(model_base_dir)
now_str = datetime.now().isoformat()
save_dir = os.path.join(model_base_dir,
                        now_str + f"-{MODEL_NAME}")

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
trainer.fit(model, dataloader)
##
# CKPT_PATH = "/scratch/models/2021-05-25T03:15:43.684726-PerVoxelLSTM.ckpt"
# CKPT_PATH = "/scratch/models/2021-06-15T20:15:52.602236-PerVoxelLSTM.ckpt"
# CKPT_PATH = "/scratch/models/2021-06-18T05:53:49.997675-PerVoxelLSTM.ckpt"
CKPT_PATH = "/scratch/models/2021-06-18T06:32:01.442905-PerVoxelLSTM.ckpt"
best_model = PerVoxel.load_from_checkpoint(CKPT_PATH, net=LSTM(batch_size=n_voxels*batch_size),
                                           H=H, W=W, Z=Z)
best_model = best_model.cuda()
# model = best_model
##
# best_model = model.cuda()
##
with torch.no_grad():
    x, y = dataset[0]
    x = torch.from_numpy(x[:, None]).repeat(1, 4, 1, 1, 1, 1)
    x = x.cuda()
    clean = best_model(x)
    clean = clean[:, 0].cpu().numpy()
##
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
im0 = axes[0].imshow(exposure.adjust_gamma(y[0, 5], 0.2))
# im = axes[1].imshow(clean[0, 5])
im = axes[1].imshow(exposure.adjust_gamma(clean[0, 5],0.2))
fig.colorbar(im)
##
full_dataset = VolumeSeq(green[:, None], [],
                    n_time_steps=n_time_steps)
inference_batch_size = 4  # even 8 OOM...?
dataloader = torch.utils.data.DataLoader(
    full_dataset, batch_size=inference_batch_size, collate_fn=rnn_collate_fn, shuffle=False,
    num_workers=1, pin_memory=True)

tyh5_path = tif_dir + ".ty.h5"
out_dataset = "/imaging/per_pixel_lstm_denoised_maybe_longer_time"
with tables.open_file(tyh5_path, 'a') as tyh5:
    if out_dataset in tyh5:
        # raise ValueError(f"dset ({out_dataset}) already exists")
        tyh5.remove_node(out_dataset)
    dset_filter = tables.filters.Filters(complevel=0, complib='blosc:zstd')
    group, dset_name = os.path.split(out_dataset)
    parent, name = os.path.split(group)
    if not group in tyh5:
        h5_group = tyh5.create_group(parent, name, createparents=True)
    else:
        h5_group = tyh5.root[group]

    dtype = np.dtype("uint16")
    dset = tyh5.create_carray(group, dset_name,
                              tables.Atom.from_dtype(dtype), shape=green[:, None].shape,
                              filters=dset_filter)
    with torch.no_grad():
        for b, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
            x, y = batch_data
            x = x.cuda()
            clean = best_model(x)
            # remove channel
            # clean = clean[:, 0].cpu().numpy()
            clean = clean.cpu().numpy()
            s = n_time_steps + b*inference_batch_size
            dset[s:s+inference_batch_size] = clean
##
out_dataset = "/imaging/raw"
with tables.open_file(tyh5_path, 'a') as tyh5:
    if out_dataset in tyh5:
        raise ValueError(f"dset ({out_dataset}) already exists")
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
                              filters=dset_filter)
    with torch.no_grad():
        for b, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
            x, y = batch_data
            s = n_time_steps + b*inference_batch_size
            dset[s:s+inference_batch_size] = y.numpy()


##
# TODO
raise(NotImplementedError())
pw = os.environ["POSTGRES_OPTUNA_PASSWORD"]
server = os.environ["POSTGRES_SERVER"]
port = os.environ["POSTGRES_PORT"]
user = os.environ["POSTGRES_OPTUNA_USER"]
# validate password

engine = create_engine(f'postgresql://{user}:{pw}@{server}:{port}/optuna')
with engine.connect() as connection:
    result = connection.execute("select * from trials limit 1;")
