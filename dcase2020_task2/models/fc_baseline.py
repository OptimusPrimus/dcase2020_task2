import torch.nn
from models import VAEBase
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.init as init


class FCBaseLine(torch.nn.Module, VAEBase):

    def __init__(
            self,
            input_shape,
            reconstruction_loss,
            prior,
            normalize_fun
    ):
        super().__init__()

        self.input_shape = input_shape
        self.prior = prior
        self.reconstruction = reconstruction_loss
        self.normalize_fun = normalize_fun

        self.encoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(np.prod(input_shape)),
            torch.nn.Linear(np.prod(input_shape), 128, bias=False),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 128, bias=False),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 128, bias=False),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 128, bias=False),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, prior.latent_size, bias=False),
            torch.nn.BatchNorm1d(prior.latent_size),
            torch.nn.ReLU(True)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(prior.latent_size, 128, bias=False),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 128, bias=False),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 128, bias=False),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 128, bias=False),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, np.prod(input_shape))
        )

    def forward(self, batch):
        batch = self.normalize_fun(batch)
        batch = self.encode(batch)
        batch = self.prior(batch)
        batch = self.decode(batch)
        return batch

    def encode(self, batch):
        x = batch['normalized_observations']
        x = x.view(x.shape[0], -1)
        batch['pre_codes'] = self.encoder(x)
        return batch

    def decode(self, batch):
        batch['pre_reconstructions'] = self.decoder(batch['codes']).view(-1, *self.input_shape)
        batch = self.reconstruction(batch)
        return batch
