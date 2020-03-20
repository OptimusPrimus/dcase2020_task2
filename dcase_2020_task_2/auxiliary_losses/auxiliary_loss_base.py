from abc import ABC, abstractmethod


class AuxiliaryLossBase(ABC):

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    @abstractmethod
    def auxiliary_loss(self, batch):
        raise NotImplementedError
