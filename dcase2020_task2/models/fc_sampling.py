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


class SamplingCRNNAE(torch.nn.Module, VAEBase):

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
                                                        # (b, 1, 40)
            torch.nn.Conv1d(1, 128, kernel_size=10),    # (b, 128, 31)
            torch.nn.ReLU(True),                        # (b, 128, 31)
            torch.nn.AdaptiveMaxPool1d(1),              # (b, 128, 1)
            Reshape((1, 128)),                          # (b, 1, 128)
            torch.nn.LSTM(128, 128, batch_first=True),  # (b, 1, 128)
            Select(0),
            Reshape((128,)),                            # (b, 128)
            torch.nn.Linear(128, prior.latent_size),    # (b, 40)
        )

        channels = input_shape[1] - 10 + 1

        self.decoder = torch.nn.Sequential(
                                                                    # (b, 40)
            torch.nn.Linear(prior.latent_size, 128),                # (b, 128)
            torch.nn.ReLU(True),                                    # (b, 128)
            Reshape((1, 128)),                                      # (b, 1, 128)
            torch.nn.LSTM(128, channels*128, batch_first=True),     # (b, 1, 128*31)
            Select(0),
            Reshape((128, channels)),                               # (b, 128, 31)
            torch.nn.ConvTranspose1d(128, 1, 10)                    # (b, 1, 40)
        )

    def forward(self, batch):
        batch = self.encode(batch)
        batch = self.prior(batch)
        batch = self.decode(batch)
        return batch

    def encode(self, batch):
        x = batch['observations']
        x = x.view(x.shape[0], 1, x.shape[2])
        batch['pre_codes'] = self.encoder(x)
        return batch

    def decode(self, batch):
        x = self.decoder(batch['codes'])
        batch['pre_reconstructions'] = x.view(-1, *self.input_shape)
        batch = self.reconstruction(batch)
        return batch


class Reshape(torch.nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)


class Transpose(torch.nn.Module):

    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Select(torch.nn.Module):

    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]


