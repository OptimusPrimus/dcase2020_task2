from typing import Union, Sequence, Dict, NoReturn
from abc import ABC, abstractmethod
from dcase2020_task2.experiments import BaseExperiment


class BaseTrainer(ABC):

    @abstractmethod
    def fit(self, experiment: BaseExperiment) -> NoReturn:
        raise NotImplementedError()

    @abstractmethod
    def test(self, experiment: BaseExperiment) -> NoReturn:
        raise NotImplementedError()
