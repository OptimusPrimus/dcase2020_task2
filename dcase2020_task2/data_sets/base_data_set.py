from abc import ABC, abstractmethod, abstractproperty
from typing import Any


class BaseDataSet(ABC):

    @property
    @abstractmethod
    def observation_shape(self) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError