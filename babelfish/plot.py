import torch as T
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
# TODO switch to moviepy
import skvideo.io


def plot_model_vs_real(model,data):
    plt.figure(figsize=(30,15))

    with T.no_grad():
        for i in range(4):
            time = np.random.randint(len(data))
            z = np.random.randint(nZ)
            X, Y = data[time]
            X, X_shock, X_tail = (X["brain"], X["shock"], X["tail_movement"])
            Y, Y_shock, Y_tail = (Y["brain"], Y["shock"], Y["tail_movement"])
            x = X[None].cuda()
            Y_shock = Y_shock[None].cuda()
            y = Y[None].cuda()
            (Y_pred, Y_pred_tail), embedding = model(x, Y_shock)
            mse_Y_pred_to_X = float(F.mse_loss(Y_pred,x[:,-1]).cpu())
            mse_Y = float(F.mse_loss(Y_pred,y[:,-1]).cpu())
            prev_loss = float(F.mse_loss(x[:,0],y[:,-1]).cpu())
#             x_zero_loss = float(F.mse_loss(x,T.zeros_like(x)).cpu())
#             y_zero_loss = float(F.mse_loss(y,T.zeros_like(y)).cpu())
            mymin = min(float(y[0,:,z].min()[0]),float(x[0,:,z].min()[0]),float(Y_pred[0,z].min()[0]))
            mymax = max(float(y[0,:,z].max()[0]),float(x[0,:,z].max()[0]),float(Y_pred[0,z].max()[0]))

            plt.subplot(4,4,i*4+1)
            plt.imshow(X[-1,z].cpu().numpy(), vmin=mymin, vmax=mymax)
            plt.title("Time="+str(time) + ", z="+str(z))

            plt.subplot(4,4,i*4+2)
            plt.imshow(Y[0,z].cpu().numpy(), vmin=mymin, vmax=mymax)
            plt.title("Time="+str(time+1) + ", z="+str(z))

            plt.subplot(4,4,i*4+4)
            plt.imshow(Y_pred[0,z].cpu().numpy(), vmin=mymin, vmax=mymax)
            plt.title("MSE: (Y_pred,Y)={:.0f}, (Y_pred,X)={:.0f}".format(mse_Y,mse_Y_pred_to_X))

def interpret(model,prev_vol, next_vol, nEmbedding, scale=1):
    "Plot prev & next frame for each latent dimension"
    plt.figure(figsize=(10,40))

    embedding = T.from_numpy(np.zeros(nEmbedding).astype(np.float32)).cuda()[None]
#     shock = T.FloatTensor([0,1])[:,None,None].cuda()
    shock = T.cuda.FloatTensor(2,1,prev_frames).zero_()
    shock[0] = T.zeros(1,prev_frames)
    shock[1] = T.ones(1,prev_frames)
    with T.no_grad():
        prev_img = model.decode(embedding)[0][0] # ignore tail

    zero_vector = prev_img[6]
    plt.subplot(1+nEmbedding,2,1)
    plt.imshow(zero_vector)
    plt.title("Zero Vector")

    for i in range(nEmbedding):
        embedding = T.from_numpy(np.eye(nEmbedding)[i].astype(np.float32)).cuda()[None]*scale
        with T.no_grad():
            prev_img = model.decode(embedding)[0][0]
        plt.subplot(1+nEmbedding,2,i*2+3)
        plt.imshow(prev_img[6])
        plt.title("Dim {}".format(i))
        plt.subplot(1+nEmbedding,2,i*2+4)
        plt.imshow(prev_img[6] - zero_vector)
        plt.title("Dim {} diff from zero".format(i))

    plt.tight_layout()


class FishDistanceData(Dataset):
    def __init__(self, imaging, distance):
        data = imaging - imaging.mean(0)
        self.data = T.from_numpy(data)
        self.distance=distance

    def __len__(self):
        return self.data.shape[0]-self.distance

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx+self.distance]

def MSEbyDist(imaging, maxdist=10, batch_size=256):
    mse = []

    with T.no_grad():
        for d in range(1,maxdist+1):
            data = FishDistanceData(imaging,d)
            dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            mse.append([])
            for batch_data in tqdm(dataloader):
                X, Y = batch_data
                X = X.cuda()
                Y = Y.cuda()
                mse[d-1].append(volume_mse(X, Y).cpu())

    mse = [T.cat(m).numpy() for m in mse]
    return mse

def plot_embedding_over_time(model,data, batch_size=64, num_workers=12):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    embeddings = []
    logvars = []
    model.eval()
    for batch_data in dataloader:
        X, Y = batch_data
        X, X_shock, X_tail = (X["brain"], X["shock"], X["tail_movement"])
        Y, Y_shock, Y_tail = (Y["brain"], Y["shock"], Y["tail_movement"])
        with T.no_grad():
            embedding, logvar, _ = model.encode(X.cuda())
        embeddings.append(embedding.cpu().numpy())
        logvars.append(logvar.cpu().numpy())
    model.train()
    embeddings = np.vstack(embeddings)
    logvars = np.vstack(logvars)
    nEmbeddings = embeddings.shape[1]
    half = int(np.ceil(nEmbeddings / 2))

    plt.figure(figsize=(15,20))
    plt.subplot(4,1,1)
    plt.plot(embeddings[:,0:half])
    plt.title("Embeddings over time")
    plt.xlabel("Time")
    plt.legend(np.arange(half))
    plt.subplot(4,1,2)
    plt.plot(embeddings[:,half:])
    plt.title("Embeddings over time")
    plt.xlabel("Time")
    plt.legend(np.arange(half,nEmbeddings))

    plt.subplot(4,1,3)
    plt.plot(logvars[:,0:half])
    plt.title("Logvars over time")
    plt.xlabel("Time")
    plt.legend(np.arange(half))
    plt.subplot(4,1,4)
    plt.plot(logvars[:,half:])
    plt.title("Logvars over time")
    plt.xlabel("Time")
    plt.legend(np.arange(half,nEmbeddings))

    plt.tight_layout()
    return embeddings

def scale_for_vid(arr):
    return ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')


def scale_for_vid(arr, mymin, mymax):
    return ((arr - mymin) * (1/(mymax - mymin)) * 255).astype('uint8')

def scale_for_vid(arr, mymin, mymax):
    return ((arr - mymin) * (1/(mymax - mymin)) * 255).astype('uint8')

def makePredVideo(model, data, batch_size=32, num_workers=12, name="test"):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    writer = skvideo.io.FFmpegWriter(name+".mp4",outputdict={
        '-b': '30000000', '-vcodec': 'libx264'})
    mymax = float(T.cat([data[i][0]['brain'] for i in np.arange(len(data))]).max())
    mymin = float(T.cat([data[i][0]['brain'] for i in np.arange(len(data))]).min())
    for batch_data in dataloader:
        X, Y = batch_data
        X, X_shock, X_tail = (X["brain"], X["shock"], X["tail_movement"])
        Y, Y_shock, Y_tail = (Y["brain"], Y["shock"], Y["tail_movement"])
        with T.no_grad():
            (X_pred, X_pred_tail), (Y_pred, Y_pred_tail), _, _= model(X.cuda(),Y_shock.cuda())
        for x, x_pred, y, y_pred in zip(X,X_pred,Y,Y_pred):
            # 7th z layer
            zslice = x_pred[6]
            H = zslice.shape[0]
            W = zslice.shape[1]
            frame = np.zeros([H*2,W*3])

            frame[:H, :W] = y[0,6]
            frame[:H, W:(2*W)] = y[-1,6] - y[0,6]
            frame[:H, (2*W):] = y[-1,6]
            frame[H:, :W] = x_pred[6]
            frame[H:, W:(2*W)] = y_pred[6] - y[0,6].cuda() #x_pred[6]
            frame[H:, (2*W):] = y_pred[6]
            writer.writeFrame(scale_for_vid(frame,mymin,mymax))
    writer.close()
    return frame
