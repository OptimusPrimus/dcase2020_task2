import torch.nn
from dcase2020_task2.models import VAEBase  #
from dcase2020_task2.priors import NoPrior
import numpy as np
import torch

from dcase2020_task2.models.custom import activation_dict, init_weights


class FCNN(torch.nn.Module):

    def __init__(
            self,
            input_shape,
            hidden_size=256,
            num_hidden=3,
            num_outputs=1,
            activation='relu',
            batch_norm=False
    ):
        super().__init__()

        activation_fn = activation_dict[activation]
        self.input_shape = input_shape

        sizes = [np.prod(input_shape)] + [hidden_size // (2**l) for l in range(num_hidden)] + [num_outputs]
        layers = []
        for i, o in zip(sizes[:-1], sizes[1:]):
            layers.append(torch.nn.Linear(i, o))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(o))
            layers.append(activation_fn())

        _ = layers.pop()
        if batch_norm:
            _ = layers.pop()

        self.clf = torch.nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, batch):
        x = batch['observations']
        x = x.view(x.shape[0], -1)
        batch['scores'] = self.clf(x)
        return batch