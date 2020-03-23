import torch.nn
from models import VAEBase
import numpy as np
import torch


class SamplingFCAE(torch.nn.Module, VAEBase):

    def __init__(
            self,
            input_shape,
            reconstruction_loss,
            prior
    ):
        super().__init__()

        self.input_shape = input_shape
        self.prior = prior
        self.reconstruction = reconstruction_loss

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(np.prod(input_shape), 512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, prior.latent_size),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(prior.latent_size, 512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, np.prod(input_shape)),
        )

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


class SamplingFCGenerator(torch.nn.Module, VAEBase):

    def __init__(
            self,
            input_shape,
            reconstruction_loss,
            prior,
            encoder=None,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.prior = prior
        self.reconstruction = reconstruction_loss

        if encoder:
            self.encoder = encoder
        else:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(np.prod(input_shape), 512),
                torch.nn.ReLU(True),
                torch.nn.Linear(512, 512),
                torch.nn.ReLU(True),
                torch.nn.Linear(512, prior.latent_size),
            )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(prior.latent_size, 512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, np.prod(input_shape)),
        )

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