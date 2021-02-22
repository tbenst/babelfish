"""DataLoader utils for generic Zebrafish datasets"""
import os

import h5py
import hdf5plugin
import numpy as np
import torch
from torch.utils.data import Dataset


class ZebraFishData(Dataset):
    """
    Parameters
    ----------
    data: h5py.Group
        Should have keys:
            'imaging': {
                'raw': CArray TCZHW (1024x1024 or 512x512),
                'small' CArray TCZHW (256x256),
            'stim': {
                'masks': {
                    'raw': Array NZHW (same HW as 'imaging/raw')
                    'small': Array NZHW (same HW as 'imaging/small')
                'stim_used_at_each_timestep': Array T
            }

        (N: number of unique stimuli)

    data_key: str
        Key used for both raw data and stim data
    
    mode: str \in ('index_map', 'stim_only')

    Notes
    -----
    For now just returning stim data whenever it's available, could make this optional
    """
    IMAGING_KEY = 'imaging'
    STIM_MASK_KEY = 'stim/masks'
    STIM_ORDER_KEY = 'stim/stim_used_at_each_timestep'

    def __init__(
        self,
        data,
        data_key='small',
        mode='all',
        index_map=None,
        stim_start_end=None,
        prev_frames=2,
        next_frames=1,
        return_stim=True,
        dtype=np.float32
    ):
        if self.IMAGING_KEY not in data:
            raise ValueError('h5py does not contain {} field'.format(self.IMAGING_KEY))

        self.raw_data_key = os.path.join(self.IMAGING_KEY, data_key)
        if self.raw_data_key not in data:
            raise ValueError('h5py does not contain {} field'.format(data_key))

        self.stim_mask_key = os.path.join(self.STIM_MASK_KEY, data_key)  # e.g. 'stim/masks/small'
        if return_stim and (self.stim_mask_key not in data or self.STIM_ORDER_KEY not in data):
            raise ValueError('h5py does not contain both {} and {}'.format(self.STIM_MASK_KEY, self.STIM_ORDER_KEY))

        if mode == 'all':
            if index_map is None:
                raise ValueError('Please provide an index_map in mode="{}"'.format(mode))
        elif mode == 'stim_only':
            if stim_start_end is None:
                raise ValueError('Please provide stim_start_end in mode="{}"'.format(mode))
            assert (
                isinstance(stim_start_end, list)
                and all([isinstance(_t, tuple) and len(_t) == 2 for _t in stim_start_end])
            )
        else:
            raise ValueError('Mode {} not supported'.format(mode))
        self.mode = mode
        self.index_map = index_map
        self.stim_start_end = stim_start_end

        # TODO(allan.raventos): construct index map here if not passed in (avoids if/else everywhere)
        self.index_map = index_map

        self.ZHW = data[self.raw_data_key].shape[2:]
        self.data = data

        self.prev_frames = prev_frames
        self.next_frames = next_frames

        self.return_stim = return_stim

        self.dtype = dtype

    @staticmethod
    def get_stim_start_and_end_indices(data, stim_type=None):
        """Gets stim start and end indices from dataset as a static method so as to compute splits in an external method

        data: h5py.Group

        stim_type:
            If None then return indices for all stim types
        """
        assert stim_type is None or (isinstance(stim_type, int) and stim_type > 0)
        assert ZebraFishData.STIM_ORDER_KEY in data

        # Stim used at each timestep. 0 denotes no stim, so change from 0 to non-zero denotes start of a stim
        stim_per_timestep = [int(_stim_type) for _stim_type in data[ZebraFishData.STIM_ORDER_KEY]]

        # Starting and ending with no stim is assumed
        assert stim_per_timestep[0] == 0 and stim_per_timestep[-1] == 0

        stim_start_indices, stim_end_indices = [], []
        for idx in range(1, len(stim_per_timestep) - 1):
            idx_is_stim = (stim_type is None and stim_per_timestep[idx] > 0 or stim_per_timestep[idx] == stim_type)
            if stim_per_timestep[idx - 1] == 0 and idx_is_stim:
                stim_start_indices.append(idx)
            elif idx_is_stim and stim_per_timestep[idx + 1] == 0:
                stim_end_indices.append(idx)

        assert len(stim_start_indices) == len(stim_end_indices)
        print('Number of stims in this dataset: {}'.format(len(stim_start_indices)))

        return stim_start_indices, stim_end_indices

    def __len__(self):
        if self.mode == 'all':
            return len(self.index_map)
        elif self.mode == 'stim_only':
            return len(self.stim_start_end)

    def __getitem__(self, i):
        "X[0]==X_i, X[1]==X_i-1, Y[0]==Y_i+1, Y[1]==Y_i+2"
        if self.mode == 'all':
            last_X_idx = self.index_map[i]
            first_Y_idx = last_X_idx + 1
        elif self.mode == 'stim_only':
            stim_start, stim_end = self.stim_start_end[i]
            last_X_idx = stim_start - 1  # index right before stim
            first_Y_idx = stim_end + 1  # index right after stim

        # Populate previous frames X
        X = {'brain': [], 'indices': []}
        if self.return_stim:
            X['stim'] = []

        # Previous frame logic is identical for mode='all' and mode='stim_only'
        for i in reversed(range(self.prev_frames)):
            ix = last_X_idx - i  # ix is context index
            datum = torch.tensor(self.data[self.raw_data_key][ix].astype(self.dtype))
            X['brain'].append(datum)

            if self.return_stim:
                stim_used = int(self.data[self.STIM_ORDER_KEY][ix])
                if stim_used == 0:
                    stim = torch.tensor(np.zeros(self.ZHW, dtype=self.dtype))
                else:
                    # -1 is critical (since 0 is reserved for no stim at that index)
                    stim = torch.tensor(self.data[self.stim_mask_key][stim_used - 1].astype(self.dtype))

                X['stim'].append(stim)

            X['indices'].append(ix)

        # Populate future frames Y
        Y = {'brain': [], 'indices': []}
        for i in range(self.next_frames):
            ix = first_Y_idx + i
            datum = torch.tensor(self.data[self.raw_data_key][ix].astype(self.dtype))
            Y['brain'].append(datum)

            Y['indices'].append(ix)

        X['brain'] = torch.stack(X['brain'], 0)
        # post-collate batch will have indices of batch i in row i
        X['indices'] = torch.tensor(X['indices'])
        if self.return_stim:
            X['stim'] = torch.stack(X['stim'], 0)
        Y['brain'] = torch.stack(Y['brain'])
        Y['indices'] = torch.tensor(Y['indices'])

        return X, Y


def test_stim_only(tyh5, prev_frames, next_frames):
    from babelfish.babelfish.helpers.helpers import train_test_split_stim
    stim_start_indices, stim_end_indices = ZebraFishData.get_stim_start_and_end_indices(tyh5, stim_type=1)
    train_start_end, test_start_end = train_test_split_stim(stim_start_indices, stim_end_indices)

    train_data = ZebraFishData(
        tyh5,
        data_key='small',
        mode='stim_only',
        stim_start_end=train_start_end,
        prev_frames=prev_frames,
        next_frames=next_frames,
        return_stim=True
    )
    # print('train')
    # print('-' * 80)
    # print(len(train_data))
    # X, Y = train_data[0]
    # print('X has keys: {}. Y has keys: {}'.format(X.keys(), Y.keys()))
    # print('X: raw: {}, stim: {}, indices: {}'.format(X['brain'].shape, X['stim'].shape, X['indices']))
    # print('Y: raw: {}, indices: {}'.format(Y['brain'].shape, Y['indices']))

    # Check stim durations
    for datum in train_data:
        X, Y = datum
        diff = Y['indices'][0] - X['indices'][-1] - 2
        assert diff in (6, 7)

    # Iterate using dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(train_data, batch_size=2, num_workers=4)
    for batch in dataloader:
        X, Y = batch
        print(X['brain'].shape[0])


def test_all(tyh5, prev_frames, next_frames):
    from babelfish.babelfish.helpers.helpers import train_test_split

    # Just to know number of timestamps
    imaging = tyh5['/imaging/small']
    assert len(imaging.shape) == 5
    T = imaging.shape[0]

    tvt_split = train_test_split(T, prev_frames=prev_frames, next_frames=next_frames, n_per_sample=10, nchunks=20)
    train_data = ZebraFishData(
        tyh5,
        data_key='raw',
        mode='all',
        index_map=tvt_split['train'],
        prev_frames=prev_frames,
        next_frames=next_frames,
        return_stim=True
    )


if __name__ == '__main__':
    tyh5_path = '/home/allan/workspace/lensman/TSeries-lrhab_raphe_40trial-045.ty.h5'
    prev_fr = 5
    next_fr = 5
    with h5py.File(tyh5_path, 'r', swmr=True) as tyh5:
        # test_all(tyh5, prev_fr, next_fr)
        test_stim_only(tyh5, prev_fr, next_fr)
