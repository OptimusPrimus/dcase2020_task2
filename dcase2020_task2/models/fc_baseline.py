import torch.nn
from models import VAEBase
import numpy as np
import torch


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)


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
            # 1
            torch.nn.Linear(np.prod(input_shape), 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(True),
            # 2
            torch.nn.Linear(128, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(True),
            # 3
            torch.nn.Linear(128, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(True),
            # 4
            torch.nn.Linear(128, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(True),
            # bn
            torch.nn.Linear(128, prior.input_size),
            torch.nn.BatchNorm1d(prior.input_size),
            torch.nn.ReLU(True)
        )

        self.decoder = torch.nn.Sequential(
            # 5
            torch.nn.Linear(prior.latent_size, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(True),
            # 6
            torch.nn.Linear(128, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(True),
            # 7
            torch.nn.Linear(128, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(True),
            # 8
            torch.nn.Linear(128, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(True),
            # out
            torch.nn.Linear(128, np.prod(input_shape))
        )

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
            # 1
            torch.nn.Linear(np.prod(input_shape), 128),
            torch.nn.ReLU(True),
            # 2
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(True),
            # 3
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(True),
            # 4
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(True),
            # bn
            torch.nn.Linear(128, 1)
        )

        self.apply(init_weights)

    def forward(self, batch):
        x = batch['observations']
        x = x.view(x.shape[0], -1)
        batch['scores'] = self.classifier(x)
        return batch
