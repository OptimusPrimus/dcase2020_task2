from typing import NoReturn
from abc import ABC, abstractmethod
import torch
from dcase2020_task2.experiments.parser import create_objects_from_config
import copy
import os
from pathlib import Path

class BaseExperiment(ABC, torch.nn.Module):

    def __init__(self, configuration_dict):
        super(BaseExperiment, self).__init__()
        self.configuration_dict = copy.deepcopy(configuration_dict)
        self.objects = create_objects_from_config(configuration_dict)

        for k in ['machine_type', 'machine_id', 'trainer', 'data_set', 'log_path', 'batch_size', 'num_workers']:
            assert k in self.objects

        Path(self.configuration_dict['log_path']).mkdir(parents=True, exist_ok=True)

        if self.objects.get('deterministic'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.machine_type = self.objects['machine_type']
        self.machine_id = self.objects['machine_id']
        self.trainer = self.objects['trainer']

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

    def configure_optimizers(self):
        optimizers = [
            self.objects['optimizer']
        ]

        lr_schedulers = [
            self.objects['lr_scheduler']
        ]

        return optimizers, lr_schedulers

    def train_dataloader(self):
        dl = torch.utils.data.DataLoader(
            self.objects['data_set'].training_data_set(),
            batch_size=self.objects['batch_size'],
            shuffle=True,
            num_workers=self.objects['num_workers'],
            drop_last=False
        )
        return dl

    def val_dataloader(self):
        dl = torch.utils.data.DataLoader(
            self.objects['data_set'].validation_data_set(),
            batch_size=self.objects['batch_size'],
            shuffle=False,
            num_workers=self.objects['num_workers']
        )
        return dl

    def test_dataloader(self):
        dl = torch.utils.data.DataLoader(
            self.objects['data_set'].validation_data_set(),
            batch_size=self.objects['batch_size'],
            shuffle=False,
            num_workers=self.objects['num_workers']
        )
        return dl

    def run(self):
        self.trainer.fit(self)
        self.trainer.test(self)
        self.trainer.save_checkpoint(os.path.join(self.objects['log_path'], "model.ckpt"))
        return self.result
