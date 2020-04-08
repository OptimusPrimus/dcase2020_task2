import torch.nn
from models import VAEBase
import numpy as np
import torch


class BaselineCNN(torch.nn.Module, VAEBase):

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
            torch.nn.Conv2d(input_shape[0], 32, (3, 3), padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d((2, 1)),

            torch.nn.Conv2d(32, 64, (3, 3), padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d((2, 1)),

            torch.nn.Conv2d(64, 128, (3, 3), padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d((2, 1)),

            torch.nn.Conv2d(128, prior.latent_size, (3, 3), padding=1),
            torch.nn.ReLU(True),

            torch.nn.AdaptiveAvgPool2d((1, 1))
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(prior.latent_size, prior.latent_size, (input_shape[-2] // 8, input_shape[-1])),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(prior.latent_size, 128, (3, 3), stride=(2, 1), padding=(1, 1), output_padding=(1, 0)),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(128, 64, (3, 3), stride=(2, 1), padding=(1, 1), output_padding=(1, 0)),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(64, 32, (3, 3), stride=(2, 1), padding=(1, 1), output_padding=(1, 0)),
            torch.nn.ReLU(True),

            torch.nn.Conv2d(32, input_shape[0], (3, 3), padding=1)
        )

    def forward(self, batch):
        batch = self.encode(batch)
        batch = self.prior(batch)
        batch = self.decode(batch)
        return batch

    def encode(self, batch):
        x = batch['observations']
        batch['pre_codes'] = self.encoder(x).view(x.shape[0], -1)
        return batch

    def decode(self, batch):
        batch['pre_reconstructions'] = self.decoder(batch['codes'].view(-1, self.prior.latent_size, 1, 1))
        batch = self.reconstruction(batch)
        return batch

'''
from priors.no_prior import NoPrior
from losses.mse_loss import MSE
import torch

input_shape = (1, 128, 5)

prior = NoPrior(latent_size=8)
mse = MSE()

cnn = BaselineCNN(input_shape, mse, prior)

input = {
    'observations':torch.ones(1, *input_shape)
}

cnn(input)
'''