import torch
import abc


class ReconstructionBase(torch.nn.Module, abc.ABC):

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    @abc.abstractmethod
    def loss(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, batch):
        raise NotImplementedError
