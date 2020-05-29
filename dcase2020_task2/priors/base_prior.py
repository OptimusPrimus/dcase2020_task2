from abc import ABC, abstractmethod
import torch


class PriorBase(ABC, torch.nn.Module):

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def input_size(self):
        """ number of activations that go into the prior, e.g. 2 * latent_size for Gaussian prior """
        raise NotImplementedError

    @property
    @abstractmethod
    def latent_size(self):
        """ number latent dimensions """
        raise NotImplementedError


    @abstractmethod
    def forward(self, pre_prior):
        """ sampling, etc, ... """
        raise NotImplementedError