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
            cond_label_size=None,
            **kwargs
    ):
        """
        Args:
            TODO
        """
        super().__init__()

        self.input_shape = input_shape
        self.reconstruction = reconstruction

        input_degree = torch.arange(int(np.prod(input_shape))).view(input_shape).transpose(2, 1).reshape(-1)
        # input_degree = torch.arange(int(np.prod(input_shape)))

        # create masks
        # use natural order as input order
        masks, self.input_degrees = create_masks(
            int(np.prod(input_shape)),
            hidden_size,
            num_hidden,
            input_degrees=input_degree,
            input_order='sequential'
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
        batch_size = batch['observations'].shape[0]

        x = batch['observations'].reshape(batch_size, -1)
        y = batch.get('y', None)

        m, loga = self.net(self.input_layer(x, y)).chunk(chunks=2, dim=1)

        # this guys should be normally distributed....
        u = (x - m) * torch.exp(-loga)

        # MAF eq 5
        batch['u'] = u
        batch['log_abs_det_jacobian'] = - loga

        batch['reconstruction'] = m.reshape(batch_size, *self.input_shape)

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
