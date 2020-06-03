import torch.nn
from dcase2020_task2.models import VAEBase#
from dcase2020_task2.priors import NoPrior
import numpy as np
import torch

from dcase2020_task2.models.custom import activation_dict, init_weights


class AE(torch.nn.Module, VAEBase):

    def __init__(
            self,
            input_shape,
            reconstruction_loss,
            prior=NoPrior(latent_size=8),
            hidden_size=128,
            num_hidden=3,
            activation='relu',
            batch_norm=False
    ):
        super().__init__()

        activation_fn = activation_dict[activation]

        self.input_shape = input_shape
        self.prior = prior
        self.reconstruction = reconstruction_loss

        # encoder sizes/ layers
        sizes = [np.prod(input_shape) ] + [hidden_size] * num_hidden + [prior.input_size]
        encoder_layers = []
        for i, o in zip(sizes[:-1], sizes[1:]):
            encoder_layers.append(torch.nn.Linear(i, o))
            if batch_norm:
                encoder_layers.append(torch.nn.BatchNorm1d(o))
            encoder_layers.append(activation_fn())

        # symetric decoder sizes/ layers
        sizes = sizes[::-1]
        decoder_layers = []
        for i, o in zip(sizes[:-1], sizes[1:]):
            decoder_layers.append(torch.nn.Linear(i, o))
            if batch_norm:
                decoder_layers.append(torch.nn.BatchNorm1d(o))
            decoder_layers.append(activation_fn())
        # remove last relu
        _ = decoder_layers.pop()

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.decoder = torch.nn.Sequential(*decoder_layers)
        self.apply(init_weights)

    def forward(self, batch):
        batch = self.encode(batch)
        batch = self.prior(batch)
        batch = self.decode(batch)
        return batch

    def encode(self, batch):
        x = batch['observations']
        x = x.view(x.shape[0], -1)
        batch['pre_codes'] = self.encoder(x)
        return batch

    def decode(self, batch):
        batch['pre_reconstructions'] = self.decoder(batch['codes']).view(-1, *self.input_shape)
        batch = self.reconstruction(batch)
        return batch
