from __future__ import print_function, division
import numpy as np
import torch as T
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List

# updated 2019/12/15
class ZebraFishData(Dataset):
    "B x nFrames x Z x H x W"
    def __init__(self, imaging, auxiliary={},
                 index_map=None, prev_frames=2, next_frames=1, dtype=np.float32):
        # use channel for future / prev frames
        self.data = imaging
        self.prev_frames = prev_frames
        self.next_frames = next_frames
        self.auxiliary = auxiliary
        self.aux_vars = list(auxiliary.keys())
        self.index_map = index_map
        self.dtype = dtype

    def __len__(self):
        if self.index_map:
            return len(self.index_map)
        else:
            return self.data.shape[0]-self.prev_frames - self.next_frames + 1

    def __getitem__(self, i):
        "X[0]==X_i, X[1]==X_i-1, Y[0]==Y_i+1, Y[1]==Y_i+2"
        if self.index_map:
            idx = self.index_map[i]
        else:
            idx = i + self.prev_frames - 1 # avoid wraparound
        aux = {k: [] for k in self.auxiliary.keys()}
        X = {"brain": []}
        X.update(aux)
        Y = {"brain": []}
        Y.update(aux)
        for i in reversed(range(self.prev_frames)):
            ix = idx-i
            datum = T.tensor(self.data[ix].astype(self.dtype))
            X["brain"].append(datum)
            for k,v in self.auxiliary.items():
                X[k].append(v[ix])
        for i in range(1,self.next_frames+1):
            ix = idx+i
            datum = T.tensor(self.data[ix].astype(self.dtype))
            Y["brain"].append(datum)
            for k,v in self.auxiliary.items():
                Y[k].append(v[ix])
        X = {k: T.stack(v,0) for k,v in X.items()}
        Y = {k: T.stack(v,0) for k,v in Y.items()}
        return X, Y


class ZebraFishDataRNA(Dataset):
    "B x nFrames x Z x H x W"
    def __init__(self, imaging, structural, shocks, tail_movements,
                 index_map=None, prev_frames=2, next_frames=1):
        data = imaging - imaging.mean(0)
        # use channel for future / prev frames
        self.data = T.from_numpy(data)
        self.prev_frames = prev_frames
        self.next_frames = next_frames
        self.shocks = shocks
        self.tail_movements = tail_movements
        self.index_map = index_map
        self.structural = structural

    def __len__(self):
        if self.index_map:
            return len(self.index_map)
        else:
            return self.data.shape[0]-self.prev_frames - self.next_frames + 1

    def __getitem__(self, i):
        "X[0]==X_i, X[1]==X_i-1, Y[0]==Y_i+1, Y[1]==Y_i+2"
        if self.index_map:
            idx = self.index_map[i]
        else:
            idx = i + self.prev_frames - 1 # avoid wraparound
        X = {"brain": [], "shock": [], "tail_movement": []}
        Y = {"brain": [], "shock": [], "tail_movement": []}
        for i in reversed(range(self.prev_frames)):
            ix = idx-i
            X["brain"].append(self.data[ix])
            X["shock"].append(self.shocks[ix])
            X["tail_movement"].append(self.tail_movements[ix])
        for i in range(1,self.next_frames+1):
            ix = idx+i
            Y["brain"].append(self.data[ix])
            Y["shock"].append(self.shocks[ix])
            Y["tail_movement"].append(self.tail_movements[ix])
        for s in structural:
            X["brain"].append(s)
        X = {k: T.stack(v,0) for k,v in X.items()}
        Y = {k: T.stack(v,0) for k,v in Y.items()}
        return X, Y

class ZebraFishDataCaiman(Dataset):
    """B x nFrames x Z x H x W. Given Caiman, predict raw
    """
    def __init__(self, denoised_imaging, raw_imaging, structural, shocks, tail_movements,
                 index_map=None, prev_frames=2, next_frames=1):
        # assumes already mean subtracted
        assert np.all(denoised_imaging.shape == raw_imaging.shape)
        # use channel for future / prev frames
        self.denoised_data = denoised_imaging
        self.raw_data = raw_imaging
        self.prev_frames = prev_frames
        self.next_frames = next_frames
        self.shocks = shocks
        self.tail_movements = tail_movements
        self.index_map = index_map
        self.structural = structural

    def __len__(self):
        if self.index_map:
            return len(self.index_map)
        else:
            return self.denoised_data.shape[0]-self.prev_frames - self.next_frames + 1

    def __getitem__(self, i):
        "X[0]==X_i, X[1]==X_i-1, Y[0]==Y_i+1, Y[1]==Y_i+2"
        if self.index_map:
            idx = self.index_map[i]
        else:
            idx = i + self.prev_frames - 1 # avoid wraparound
        X = {"brain": [], "shock": [], "tail_movement": []}
        Y = {"brain": [], "shock": [], "tail_movement": []}
        for i in reversed(range(self.prev_frames)):
            ix = idx-i
            X["brain"].append(T.from_numpy(self.denoised_data[ix]))
            X["shock"].append(self.shocks[ix])
            X["tail_movement"].append(self.tail_movements[ix])
        for i in range(1,self.next_frames+1):
            ix = idx+i
            Y["brain"].append(T.from_numpy(self.raw_data[ix]))
            Y["shock"].append(self.shocks[ix])
            Y["tail_movement"].append(self.tail_movements[ix])
        if not self.structural is None:
            for s in self.structural:
                X["brain"].append(s)
        X = {k: T.stack(v,0) for k,v in X.items()}
        Y = {k: T.stack(v,0) for k,v in Y.items()}
        return X, Y


def rnn_collate_fn(tensors):
    "Use for LSTM where need to have tuple."
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
                 n_time_steps: int = 10, divide_by=512):
        self.n_time_steps = n_time_steps
        self.volumes = volumes
        self.divide_by = divide_by

        n_exclude = len(exclude_frames_from_y)
        self.length = self.volumes.shape[0] - self.n_time_steps - n_exclude
        self.idx_map = make_index_map(self.length, exclude_frames_from_y)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        last_idx = i+self.n_time_steps
        # we add batch singleton dimension
        x = self.volumes[i:last_idx].astype(np.float32)
        y = self.volumes[last_idx].astype(np.float32)
        # x /= 8192
        # y /= 8192
        x /= self.divide_by
        y /= self.divide_by
        return (x, y)
