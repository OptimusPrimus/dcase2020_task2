from abc import ABC, abstractmethod


class VAEBase(ABC):

    @abstractmethod
    def __init__(self, input_shape, reconstruction_loss, prior):
        raise NotImplementedError

    @abstractmethod
    def encode(self, x):
        raise NotImplementedError

    @abstractmethod
    def decode(self, x):
        raise NotImplementedError


class ClassifierBase(ABC):

    @abstractmethod
    def __init__(self, input_shape):
        raise NotImplementedError
