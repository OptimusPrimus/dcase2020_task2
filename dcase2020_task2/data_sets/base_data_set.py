from abc import ABC, abstractmethod, abstractproperty
from typing import Any


class BaseDataSet(ABC):

    @property
    @abstractmethod
    def observation_shape(self) -> tuple:
        raise NotImplementedError

    @property
    @abstractmethod
    def training_data_set(self, type, id):
        raise NotImplementedError

    @property
    @abstractmethod
    def validation_data_set(self, type, id):
        raise NotImplementedError

    @property
    @abstractmethod
    def complement_data_set(self, type, id):
        raise NotImplementedError
