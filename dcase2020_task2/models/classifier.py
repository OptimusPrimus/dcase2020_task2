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
            batch_norm=False,
            dropout_probability=0.1
    ):
        super().__init__()

        activation_fn = activation_dict[activation]
        self.input_shape = input_shape

        sizes = [np.prod(input_shape)] + [hidden_size for i in range(num_hidden)] + [num_outputs]
        layers = []
        for i, o in zip(sizes[:-1], sizes[1:]):
            layers.append(torch.nn.Linear(i, o))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(o))
            layers.append(torch.nn.Dropout(p=dropout_probability))
            layers.append(activation_fn())

        _ = layers.pop()
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


class CNN(torch.nn.Module):

    def __init__(
            self,
            input_shape,
            hidden_size=256,
            num_hidden=3,
            num_outputs=1,
            activation='relu',
            batch_norm=False,
            dropout_probability=0.1
    ):
        super().__init__()

        activation_fn = activation_dict[activation]
        self.input_shape = input_shape

        sizes = [input_shape[0]] + [hidden_size for i in range(num_hidden)] + [num_outputs]
        layers = []
        for i, o in zip(sizes[:-1], sizes[1:]):
            layers.append(torch.nn.Conv2d(i, o, kernel_size=(3, 3), stride=2, padding=(1, 1)))
            # layers.append(torch.nn.Dropout(p=dropout_probability))
            layers.append(activation_fn())

        _ = layers.pop()
        layers.append(torch.nn.AdaptiveMaxPool2d(1))
        # _ = layers.pop()


        if batch_norm:
            _ = layers.pop()

        self.clf = torch.nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, batch):
        x = batch['observations']
        batch['scores'] = self.clf(x).view(-1, 1)
        return batch