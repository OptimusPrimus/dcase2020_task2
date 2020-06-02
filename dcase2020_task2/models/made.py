import torch
import numpy as np
from torch import nn as nn, distributions as D

from dcase2020_task2.models.custom import create_masks, MaskedLinear


class MADE(nn.Module):
    def __init__(
            self,
            input_shape,
            reconstruction,
            hidden_size=4096,
            num_hidden=4,
            activation='relu',
            input_order='random',
            cond_label_size=None
    ):
        """
        Args:
            TODO
        """
        super().__init__()

        self.input_shape = input_shape
        self.reconstruction = reconstruction

        # create masks
        # use natural order as input order
        masks, self.input_degrees = create_masks(
            int(np.prod(input_shape)),
            hidden_size,
            num_hidden,
            input_order=input_order,
            input_degrees=torch.arange(int(np.prod(input_shape)))
        )

        # setup activation
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('Check activation function.')

        # construct model
        self.input_layer = MaskedLinear(np.prod(input_shape), hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [activation_fn, MaskedLinear(hidden_size, 2 * np.prod(input_shape), masks[-1].repeat(2, 1))]
        self.net = nn.Sequential(*self.net)

    def forward(self, batch):
        # MAF eq 4 -- return mean and log std
        x = batch['observations']
        x = x.view(x.shape[0], -1)
        y = batch.get('y', None)

        batch['pre_reconstructions'] = self.net(
            self.input_layer(
                x,
                y
            )
        )

        batch = self.reconstruction(batch)

        return batch

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        # MAF eq 3
        x = torch.zeros_like(u)
        # run through reverse model
        for i in self.input_degrees:
            m, loga = self.net(x, y).chunk(chunks=2, dim=1)
            x[:, i] = u[:, i] * torch.exp(loga[:, i]) + m[:, i]
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian
