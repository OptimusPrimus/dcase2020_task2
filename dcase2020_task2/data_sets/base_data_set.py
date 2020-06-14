from abc import ABC, abstractmethod, abstractproperty
from typing import Any


class BaseDataSet(ABC):

    @property
    @abstractmethod
    def observation_shape(self) -> tuple:
        raise NotImplementedError

    @property
    @abstractmethod
    def training_data_set(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def validation_data_set(self):
        raise NotImplementedError
