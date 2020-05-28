import torch
import abc


class BaseReconstruction(torch.nn.Module, abc.ABC):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, batch):
        raise NotImplementedError


class BaseLoss(torch.nn.Module, abc.ABC):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, batch_normal, batch_augmented):
        raise NotImplementedError

