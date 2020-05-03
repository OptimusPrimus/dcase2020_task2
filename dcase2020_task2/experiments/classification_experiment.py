from experiments import BaseExperiment
import pytorch_lightning as pl
import torch
from sacred import Experiment
from utils.logger import Logger
import os
import torch.utils.data
# workaround...
from sacred import SETTINGS

SETTINGS['CAPTURE_MODE'] = 'sys'
from datetime import datetime
import numpy as np


class ClassifiactionExperiment(BaseExperiment, pl.LightningModule):

    def __init__(self, configuration_dict, _run):
        super().__init__(configuration_dict)

        self.network = self.objects['model']
        self.data_set = self.objects['data_set']
        self.loss = self.objects['loss']
        self.logger_ = Logger(_run, self, self.configuration_dict, self.objects)

        # will be set before each epoch
        self.inf_data_loader = self.get_inf_data_loader(
            torch.utils.data.DataLoader(
                self.data_set.complement_data_set(self.machine_type, self.machine_id),
                batch_size=self.objects['batch_size'],
                shuffle=True,
                num_workers=self.objects['num_workers'],
                drop_last=True
            )
        )

        # experiment state variables
        self.epoch = -1
        self.step = 0
        self.result = None

    def get_inf_data_loader(self, dl):
        device = next(iter(self.network.parameters())).device
        for batch in iter(dl):
            for key in batch:
                if type(batch[key]) is torch.Tensor:
                    batch[key] = batch[key].to(device)
            yield batch

    def forward(self, batch):
        batch['epoch'] = self.epoch
        batch = self.network(batch)
        return batch

    def training_step(self, batch_normal, batch_num, optimizer_idx=0):

        if batch_num == 0 and optimizer_idx == 0:
            self.epoch += 1

        if optimizer_idx == 0:
            batch_normal = self(batch_normal)
            batch_abnormal = self(next(self.inf_data_loader))

            loss = self.loss.loss(batch_normal, batch_abnormal)

            batch_normal['loss'] = loss

            self.logger_.log_training_step(batch_normal, self.step)
            self.step += 1
        else:
            raise AttributeError

        return {
            'loss': batch_normal['loss'],
            'tqdm': {'loss': batch_normal['loss']},
        }

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

    def test_step(self, batch, batch_num):
        return self.validation_step(batch, batch_num)

    def test_end(self, outputs):
        self.result = self.logger_.log_testing(outputs)
        self.logger_.close()
        return self.result


def configuration():
    seed = 1220
    deterministic = False
    id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_path = os.path.join('..', 'experiment_logs', id)

    #####################
    # quick configuration, uses default parameters of more detailed configuration
    #####################

    machine_type = 0
    machine_id = 0

    batch_size = 512

    debug = False
    if debug:
        epochs = 1
        num_workers = 0
    else:
        epochs = 100
        num_workers = 4

    learning_rate = 1e-4
    weight_decay = 1e-4

    rho = 0.1

    feature_context = 'short'
    loss_class = 'losses.BCE'
    mse_weight = 0.0
    model_class = 'models.BaselineFCNN'

    normalize = 'all'
    normalize_raw = True

    complement = 'all'

    # TODO: change default descriptor
    descriptor = "baseline_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        model_class,
        loss_class,
        batch_size,
        learning_rate,
        weight_decay,
        normalize,
        normalize_raw,
        rho,
        feature_context,
        complement
    )

    ########################
    # detailed configurationSamplingFCAE
    ########################

    if feature_context == 'short':
        context = 5
        num_mel = 128
        n_fft = 1024
        hop_size = 512
    elif feature_context == 'long':
        context = 11
        num_mel = 40
        n_fft = 512
        hop_size = 256

    if model_class == 'models.SamplingCRNNAE':
        context = 1

    data_set = {
        'class': 'data_sets.MCMDataSet',
        'kwargs': {
            'context': context,
            'num_mel': num_mel,
            'n_fft': n_fft,
            'hop_size': hop_size,
            'normalize': normalize,
            'normalize_raw': normalize_raw,
            'complement': complement
        }
    }

    loss = {
        'class': loss_class,
        'kwargs': {
            'weight': 1.0,
            'input_shape': '@data_set.observation_shape',
            'rho': rho,
            'mse_weight': mse_weight
        }
    }

    model = {
        'class': model_class,
        'args': [
            '@data_set.observation_shape',
            '@loss'
        ]
    }

    lr_scheduler = {
        'class': 'torch.optim.lr_scheduler.StepLR',
        'args': [
            '@optimizer',
        ],
        'kwargs': {
            'step_size': 50
        }
    }

    optimizer = {
        'class': 'torch.optim.Adam',
        'args': [
            '@model.parameters()'
        ],
        'kwargs': {
            'lr': learning_rate,
            'betas': (0.9, 0.999),
            'amsgrad': False,
            'weight_decay': weight_decay,
        }
    }

    trainer = {
        'class': 'trainers.PTLTrainer',
        'kwargs': {
            'max_epochs': epochs,
            'checkpoint_callback': False,
            'logger': False,
            'early_stop_callback': False,
            'gpus': [0],
            'show_progress_bar': True,
            'progress_bar_refresh_rate': 1000
        }
    }


ex = Experiment('dcase2020_task2_classification')
cfg = ex.config(configuration)


@ex.automain
def run(_config, _run):
    experiment = ClassifiactionExperiment(_config, _run)
    return experiment.run()
