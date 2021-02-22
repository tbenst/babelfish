"RNN"
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelWiseRNNEfficient(nn.Module):
    """
    Might not be necessary to create a new class, but we want one RNN per pixel to be implemented
    efficiently and with H*W RNN's this is probably easiest through convolutions
    """
    def __init__(self, Z, H, W, len_in, len_out, n_intermediate=0):
        """
        do need to know how many need to come out
        """
        super().__init__()
        self.Z = Z
        self.H = H
        self.W = W
        self.hidden_size = 1

        self.len_in = len_in
        self.len_out = len_out
        assert self.len_in == self.len_out, 'not supported yet'
        assert n_intermediate == 0, 'not supported yet'
        self.n_intermediate = n_intermediate

        # Instantiate weights: (NOTE: skipping bias b_ih_0)
        self.W_ih_0 = nn.Parameter(torch.ones(self.Z, self.H, self.W))
        # With t=1 in and t=1 out (*and no weight reg*) this weight is irrelevant:
        self.W_hh_0 = nn.Parameter(torch.ones(self.Z, self.H, self.W))
        self.b_hh_0 = nn.Parameter(torch.zeros(self.Z, self.H, self.W))

        # Later try adding more hidden weights (so only predicting first output
        # after seeing all inputs):
        # for i in range(1, len_out):
        #     setattr(self, 'W_hh_{}'.format(i), nn.Parameter(torch.ones(self.Z, self.H, self.W)))
        #     setattr(self, 'b_hh_{}'.format(i), nn.Parameter(torch.zeros(self.Z, self.H, self.W)))

        # TODO:
        # - I think for sure want to see whole sequence before predicting any
        #   (NOTE: a bit fishy that tanh wasn't working at all)
        # - See whole input and then do easier prediction (e.g. predicting next frame should be
        #   doable as an average would be a reasonable denoising - KF performance should be achievable)

    def init_hidden(self, batch_size):
        # Maybe hidden state should be two dimensional (shouldn't need to be, but allows for flexibility)
        return torch.zeros((batch_size, self.Z, self.H, self.W)).cuda()

    def forward(self, x):
        """
        x: torch.tensor
            Shape: (B, T, Z, H, W)
                Channel dimension has been trimmed out (Green)
        """
        B, T, Z, H, W = x.shape  # might need to get rid of channels dimension
        assert B == 1, 'Broadcasting might lead to funky GD behavior'

        assert (T == self.len_in) and (Z == self.Z) and (H == self.H) and (W == self.W)

        hidden = self.init_hidden(B)

        # TODO: should use len_out and just take last outputs
        output = torch.zeros(B, self.len_out, Z, H, W).cuda()  # put on gpu

        # Printing for debugging purposes
        print('hidden sum: {}'.format(hidden.sum()))

        W_ih = self.W_ih_0
        W_hh = self.W_hh_0
        b_hh = self.b_hh_0

        for t in range(self.len_in):

            # -----------------------------------------------------
            # Need to compute:
            # h_t = tanh(W_ih * x_t + b_ih + W_hh * h_(t-1) + b_hh)
            # -----------------------------------------------------

            x_t = x[:, t, :, :, :]  # (B, Z, H, W)

            # W_ih (Z, H, W) * x_t (B, Z, H, W) --> (B, Z, H, W), make sure this propagates gradients correctly
            # hidden = W_ih * x_t + b_ih + W_hh * hidden + b_hh
            hidden = W_ih * x_t + W_hh * hidden + b_hh

            # Remove this block for first Train notebook
            print('-' * 80)
            print('Num > 0 before: {}'.format((hidden > 0).sum()))
            # hidden = F.relu(hidden)
            hidden = F.leaky_relu(hidden, negative_slope=0.01)  # leaky relu avoids 0 backpropagation to W and b
            print('Num > 0 after: {}'.format((hidden > 0).sum()))
            print('-' * 80)

            print('Hidden layer {}: {}'.format(t, hidden.sum()))

            # NOTE: this should probably change for allowing intermediate layers
            output[:, t, :, :, :] += hidden

        # for t in range(1, self.len_out):
        #     assert False

        #     # Assign last hidden to output
        #     output[:, t, :, :, :] += hidden

        #     layer_index = self.len_in + t
        #     W_hh = getattr(self, 'W_hh_{}'.format(layer_index))
        #     b_hh = getattr(self, 'b_hh_{}'.format(layer_index))

        #     hidden = W_hh * hidden + b_hh

        #     # hidden = F.relu(hidden)
        #     print('Hidden layer {}: {}'.format(layer_index, hidden.sum()))

        # output[:, self.len_out - 1, :, :, :] += hidden

        return output


class PixelWiseRNN(nn.Module):
    """
    Might not be necessary to create a new class, but we want one RNN per pixel to be implemented
    efficiently and with H*W RNN's this is probably easiest through convolutions
    """
    def __init__(self, Z, H, W, hidden_size=1):
        super().__init__()
        self.Z = Z
        self.H = H
        self.W = W
        self.hidden_size = hidden_size

        # Instantiate several RNN's, one per pixel. Could do more efficiently with matrix
        # multiplication, but first just see if this works at all (datasets will also be very tiny)
        for z, h, w in itertools.product(range(Z), range(H), range(W)):
            setattr(self, 'rnn_{}_{}_{}'.format(z, h, w), nn.RNN(1, self.hidden_size, num_layers=1))

        # NOTE: with this setup could add dummy inputs in order to get "unsynced" many-to-many

    def forward(self, x):
        """
        x: torch.tensor
            Shape: (T, Z, H, W)
        """
        # Take entry-wise product with weights, can do
        #   1. only one Z-plane first,
        #   2. separate weights for each Z,
        #   3. share weights across Z's
        B, T, Z, H, W = x.shape  # might need to get rid of channels dimension

        assert Z == self.Z and H == self.H and W == self.W

        output = torch.zeros(B, T, Z, H, W).cuda()  # put on gpu

        # (seq_len -> T, batch -> B, input_size -> 1)

        for z in range(Z):
            for h in range(H):
                for w in range(W):
                    # out, _ = self.rnns[z][h][w](x[:, z, h, w][:, None, None])
                    inp = x[:, :, z, h, w].T[:, :, None]
                    rnn = getattr(self, 'rnn_{}_{}_{}'.format(z, h, w))
                    out = rnn(inp)[0]  # (T, B, 1)
                    out = out.permute((1, 0, 2))
                    output[:, :, z, h, w] += out.squeeze()

        return output


if __name__ == '__main__':
    # dims = (10, 3, 4, 5)
    # model = PixelWiseRNN(dims[1], dims[2], dims[3])
    # print(len(list(model.parameters())))
    # x = torch.randn(dims)
    # y = model(x)

    dims = (2, 5, 3, 4, 5)
    T, Z, H, W = dims[1:]
    model = PixelWiseRNNEfficient(Z, H, W, T, T)
    print(len(list(model.named_parameters())))
    # x = torch.randn(dims)
    # y = model(x)
