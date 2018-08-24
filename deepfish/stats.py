import torch as T
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import itertools
from tqdm import tqdm
from volume import volume_mse

def sampleMSE(model,data, batch_size=96):
    prev_frames = data.prev_frames
    next_frames = data.next_frames
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    xlabels = ['X_t'] + ['X_t-{}'.format(i) for i in range(1,prev_frames)] + ['X_pred']
    ylabels = ['Y_t+{}'.format(i) for i in range(1,next_frames+1)] + ['Y_pred']
    labels = xlabels+ylabels
    mses = {"MSE({},{})".format(x,y): [] for x,y in itertools.product(labels, labels)}
    size = len(data)
    with T.no_grad():
        # stabilize running_mean and running_std of batchnorm
        niters = 0
        for batch_data in tqdm(dataloader):
            if niters >= 10:
                break
            niters += 1
            X, Y = batch_data
            X, X_shock, X_tail = (X["brain"], X["shock"], X["tail_movement"])
            Y, Y_shock, Y_tail = (Y["brain"], Y["shock"], Y["tail_movement"])
            X = X.cuda()
            Y_shock = Y_shock.cuda()
            Y = Y.cuda()
            _ = model(X, Y_shock)
    model.eval()
    for batch_data in tqdm(dataloader):
        X, Y = batch_data
        X, X_shock, X_tail = (X["brain"], X["shock"], X["tail_movement"])
        Y, Y_shock, Y_tail = (Y["brain"], Y["shock"], Y["tail_movement"])
        X = X.cuda()
        Y_shock = Y_shock.cuda()
        Y = Y.cuda()
        (X_pred, X_pred_tail), (Y_pred, Y_pred_tail), mean, logvar = model(X, Y_shock)
        xs = [X[:,i] for i in reversed(range(prev_frames))] + [X_pred]
        ys = [Y[:,i] for i in range(next_frames)] + [Y_pred]
        dat = xs+ys
        iter = itertools.product(zip(labels,dat), zip(labels,dat))
        for (xlabel, x), (ylabel, y) in iter:
            mse = volume_mse(x,y)
            mses["MSE({},{})".format(xlabel,ylabel)].append(mse)
    model.train()
    mses = {k: T.cat(v).cpu().numpy() for k,v in mses.items()}
    return mses


def sampleMSE_kSVD(model,data, batch_size=96):
    prev_frames = data.prev_frames
    next_frames = data.next_frames
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    xlabels = ['X_t'] + ['X_t-{}'.format(i) for i in range(1,prev_frames)]
    ylabels = ['Y_t+{}'.format(i) for i in range(1,next_frames+1)] + ['Y_pred']
    labels = xlabels+ylabels
    mses = {"MSE({},{})".format(x,y): [] for x,y in itertools.product(labels, labels)}
    size = len(data)
    with T.no_grad():
        # stabilize running_mean and running_std of batchnorm
        niters = 0
        for batch_data in tqdm(dataloader):
            if niters >= 10:
                break
            niters += 1
            X, Y = batch_data
            X, X_shock, X_tail = (X["brain"], X["shock"], X["tail_movement"])
            Y, Y_shock, Y_tail = (Y["brain"], Y["shock"], Y["tail_movement"])
            X = X.cuda()
            Y_shock = Y_shock.cuda()
            Y = Y.cuda()
            _ = model(X, Y_shock)
    model.eval()
    for batch_data in tqdm(dataloader):
        X, Y = batch_data
        X, X_shock, X_tail = (X["brain"], X["shock"], X["tail_movement"])
        Y, Y_shock, Y_tail = (Y["brain"], Y["shock"], Y["tail_movement"])
        X = X.cuda()
        Y_shock = Y_shock.cuda()
        Y = Y.cuda()
        (Y_pred, Y_pred_tail), embedding = model(X, Y_shock)
        xs = [X[:,i] for i in reversed(range(prev_frames))]
        ys = [Y[:,i] for i in range(next_frames)] + [Y_pred]
        dat = xs+ys
        iter = itertools.product(zip(labels,dat), zip(labels,dat))
        for (xlabel, x), (ylabel, y) in iter:
            mse = volume_mse(x,y)
            mses["MSE({},{})".format(xlabel,ylabel)].append(mse)
    model.train()
    mses = {k: T.cat(v).cpu().numpy() for k,v in mses.items()}
    return mses
