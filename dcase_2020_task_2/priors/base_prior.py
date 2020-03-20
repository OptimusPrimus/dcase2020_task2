from abc import ABC, abstractmethod
import torch


class PriorBase(ABC, torch.nn.Module):

    def __init__(self, weight=1.0, c_max=0.0, c_stop_epoch=-1):
        super().__init__()
        self.weight = weight
        self.c_max = c_max
        self.c_stop_epoch = c_stop_epoch

    @property
    @abstractmethod
    def input_size(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def latent_size(self):
        raise NotImplementedError

    @abstractmethod
    def loss(self, batch):
        raise NotImplementedError

    @abstractmethod
    def forward(self, pre_prior):
        raise NotImplementedError

    def weight_anneal(self, batch):
        if batch.get('epoch'):
            batch['c'] = self.c_max if batch['epoch'] > self.c_stop_epoch else (self.c_max / self.c_stop_epoch) * batch['epoch']
        else:
            batch['c'] = 0.0
        batch['prior_loss'] = self.weight * (batch['prior_loss'] - batch['c']).abs()
        return batch['prior_loss']