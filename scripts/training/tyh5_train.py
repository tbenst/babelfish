"""Script for training models loading from .ty.h5 format"""
import click
import h5py
import hdf5plugin  # pylint: disable=unused-import
import numpy as np
import torch

from babelfish.babelfish.data import ZebraFishData
from babelfish.babelfish.helpers.helpers import train_test_split
from babelfish.babelfish.models.deep_skip import DeepSkip as Model, train as train_model

torch.backends.cudnn.benchmark = True  # not safe for determinism
torch.backends.cudnn.deterministic = True

# PARAMETERS
half = False


@click.command()
@click.argument('tyh5_path', type=click.Path(exists=True))
@click.argument('data_key', type=str, required=False, default='small')
@click.option('--num-workers', type=int, default=16)
@click.option('--prev-frames', type=int, default=5)
@click.option('--next-frames', type=int, default=5)
@click.option('--kl-lambda', type=float, default=5e-4)  # was hardcoded to 1e-3 below earlier
@click.option('--lr', type=float, default=1e-3)
@click.option('--n-epochs', type=int, default=15)
@click.option('--n-embedding', type=int, default=2)
@click.option('--batch-size', type=int, default=8)
@click.option('--seed', type=int, default=24)
def train(
    tyh5_path, data_key, num_workers, prev_frames, next_frames, kl_lambda, lr, n_epochs, n_embedding, batch_size, seed
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    cuda = True
    with h5py.File(tyh5_path, 'r', swmr=True) as tyh5:

        # Hacky, change
        T, C, Z, H, W = tyh5['/imaging/{}'.format(data_key)].shape

        # TODO why 0.31% test?? Dig into this splitting
        tvt_split = train_test_split(T, nchunks=20)
        total_examples = sum([len(x) for x in tvt_split.values()])
        print(['{}: {} ({:.2f}%)'.format(k, len(v), 100 * len(v) / total_examples) for k, v in tvt_split.items()])

        train_data = ZebraFishData(
            tyh5,
            data_key=data_key,
            index_map=tvt_split['train'],
            prev_frames=prev_frames,
            next_frames=next_frames,
            return_stim=True
        )
        test_data = ZebraFishData(
            tyh5,
            data_key=data_key,
            index_map=tvt_split['test'],
            prev_frames=prev_frames,
            next_frames=next_frames,
            return_stim=True
        )

        tensor = torch.cuda.FloatTensor
        conv_model = Model(C, Z, H, W, n_embedding, prev_frames, next_frames, tensor=tensor)
        conv_model.cuda()

        n_params = np.sum([np.prod(x.shape) for x in conv_model.parameters()])
        print('total num params: {}', n_params)

        avg_Y_loss, avg_Y_valid_loss = train_model(
            conv_model,
            train_data,
            test_data,
            n_epochs,
            lr=lr,
            kl_lambda=kl_lambda,
            half=half,
            cuda=cuda,
            batch_size=batch_size,
            num_workers=num_workers,
            log=False
        )


if __name__ == '__main__':
    train()
