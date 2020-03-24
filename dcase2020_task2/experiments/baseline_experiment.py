from experiments import BaseExperiment
from experiments.parser import create_objects_from_config
import pytorch_lightning as pl
import torch
from sacred import Experiment
from configs.baseline_config import configuration
import copy
from utils.logger import Logger
import os
import torch.utils.data

# workaround...
from sacred import SETTINGS
SETTINGS['CAPTURE_MODE'] = 'sys'


class BaselineExperiment(pl.LightningModule, BaseExperiment):

    def __init__(self, configuration_dict, _run):
        super().__init__()

        self.configuration_dict = copy.deepcopy(configuration_dict)
        self.objects = create_objects_from_config(configuration_dict)

        if not os.path.exists(self.configuration_dict['log_path']):
            os.mkdir(self.configuration_dict['log_path'])

        self.trainer = self.objects['trainer']

        if self.objects.get('deterministic'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.auto_encoder_model = self.objects['auto_encoder_model']
        self.prior = self.objects['prior']
        self.reconstruction = self.objects['reconstruction']

        self.logger_ = Logger(_run, self, self.configuration_dict, self.objects)
        self.epoch = -1
        self.step = 0

        self.result = None

    def forward(self, batch):

        batch['epoch'] = self.epoch

        batch = self.auto_encoder_model(batch)

        reconstruction_loss = self.reconstruction.loss(batch)
        prior_loss = self.prior.loss(batch)
        loss = reconstruction_loss + prior_loss

        if self.factor:
            auxiliary_loss = self.factor.auxiliary_loss(batch)
            loss += auxiliary_loss
        batch['loss'] = loss

        return batch

    def training_step(self, batch, batch_num, optimizer_idx=0):

        if batch_num == 0 and optimizer_idx == 0:
            self.epoch += 1

        if optimizer_idx == 0:
            batch = self(batch)
            self.logger_.log_training_step(batch, self.step)
            self.step += 1
            return {
                'loss': batch['loss'],
                'tqdm': {'loss': batch['loss']},
            }
        elif optimizer_idx == 1:
            # no need to compute reconstruction loss, ...  - only need latent representation
            batch = self.auto_encoder_model(batch)
            loss = self.factor.training_loss(batch)
            self.logger_.__log_metric__('training_factor_loss', loss.item(), self.step)
            return {'loss': loss}
        else:
            raise ValueError('Too many optimizers.')

    def validation_step(self, batch, batch_num):
        self(batch)
        return {
            'targets': batch['targets'],
            'scores': batch['scores'],
            'machine_types': batch['machine_types'],
            'machine_ids': batch['machine_ids'],
            'part_numbers': batch['part_numbers'],
            'file_ids': batch['file_ids']
        }

    def validation_end(self, outputs):
        self.logger_.log_validation(outputs, self.step, self.epoch)
        return {}

    def test_step(self, batch, batch_num, *args):
        self(batch)
        return {
            'targets': batch['targets'],
            'scores': batch['scores'],
            'machine_types': batch['machine_types'],
            'machine_ids': batch['machine_ids'],
            'part_numbers': batch['part_numbers'],
            'file_ids': batch['file_ids']
        }

    def test_end(self, outputs):
        self.result = self.logger_.log_testing(outputs)
        self.logger_.close()
        return self.result

    def configure_optimizers(self):
        optimizers = [self.objects['optimizer']]
        lr_schedulers = [self.objects['lr_scheduler']]

        factor_optimizer = self.objects.get('factor_optimizer')
        factor_lr_scheduler = self.objects.get('factor_lr_scheduler')

        if factor_optimizer:
            optimizers.append(factor_optimizer)
            lr_schedulers.append(factor_lr_scheduler)

        return optimizers, lr_schedulers

    def train_dataloader(self):
        dl = torch.utils.data.DataLoader(
            self.objects['training_data_set'],
            batch_size=self.objects['batch_size'],
            shuffle=True,
            num_workers=self.objects['num_workers']
        )
        return dl

    def val_dataloader(self):
        dl = torch.utils.data.DataLoader(
            self.objects['validation_data_set'],
            batch_size=self.objects['batch_size'],
            shuffle=False,
            num_workers=self.objects['num_workers']
        )
        return dl

    def test_dataloader(self):
        dl = torch.utils.data.DataLoader(
            self.objects['validation_data_set'],
            batch_size=self.objects['batch_size'],
            shuffle=False,
            num_workers=self.objects['num_workers']
        )
        return dl

    def run(self):
        self.trainer.fit(self)
        self.trainer.test(self)
        return self.result


ex = Experiment('dcase2020_task2_baseline')
cfg = ex.config(configuration)


@ex.automain
def run(_config, _run):
    experiment = BaselineExperiment(_config, _run)
    return experiment.run()
