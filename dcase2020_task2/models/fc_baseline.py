import torch.nn
from models import VAEBase
import numpy as np
import torch


class BaselineFCAE(torch.nn.Module, VAEBase):

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
            torch.nn.Linear(np.prod(input_shape), 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, prior.latent_size),
            torch.nn.ReLU(True)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(prior.latent_size, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, np.prod(input_shape))
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


class BaselineFCNN(torch.nn.Module):

    def __init__(
            self,
            input_shape,
            reconstruction_loss
    ):
        super().__init__()

        self.input_shape = input_shape
        self.reconstruction = reconstruction_loss

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(np.prod(input_shape), 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh()
        )

    def forward(self, batch):
        x = batch['observations']
        x = x.view(x.shape[0], -1)
        batch['scores'] = self.classifier(x)
        return batch
