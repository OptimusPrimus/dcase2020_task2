import torch.nn
from models import ClassifierBase
from models.layers import Convolution2dLayer,Deconvolution2dLayer,FullyConnectedLayer
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.init as init


class Critic(torch.nn.Module, ClassifierBase):

    def __init__(self, latent_size):
        super().__init__()
        self.z_dim = latent_size
        self.net = nn.Sequential(
            nn.Linear(latent_size, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
        )

    def forward(self, z):
        return self.net(z).squeeze()
