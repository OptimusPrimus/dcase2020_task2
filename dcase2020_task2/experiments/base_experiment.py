from typing import NoReturn
from abc import ABC, abstractmethod
import torch


class BaseExperiment(ABC, torch.nn.Module):

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def training_step(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def validation_step(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def validation_end(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def test_step(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def test_end(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def train_dataloader(self):
        raise NotImplementedError()

    @abstractmethod
    def val_dataloader(self):
        raise NotImplementedError()

    @abstractmethod
    def test_dataloader(self):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()
