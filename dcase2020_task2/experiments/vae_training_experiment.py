from dcase2020_task2.experiments import BaseExperiment
from dcase2020_task2.utils.logger import Logger

from datetime import datetime
import os
import pytorch_lightning as pl
import torch
from sacred import Experiment
import torch.utils.data
# workaround...
from sacred import SETTINGS
SETTINGS['CAPTURE_MODE'] = 'sys'
import numpy as np


class VAEExperiment(BaseExperiment, pl.LightningModule):
    '''
    Reproduction of the DCASE Baseline. It is basically an Auto Encoder, the anomaly score is the reconstruction error.
    '''

    def __init__(self, configuration_dict, _run):
        super().__init__(configuration_dict)

        self.network = self.objects['auto_encoder_model']
        self.prior = self.objects['prior']
        self.reconstruction = self.objects['reconstruction']
        self.logger_ = Logger(_run, self, self.configuration_dict, self.objects)

        # experiment state variables
        self.epoch = -1
        self.step = 0
        self.result = None

    def forward(self, batch):
        batch['epoch'] = self.epoch
        batch = self.network(batch)
        return batch

    def training_step(self, batch, batch_num, optimizer_idx=0):

        if batch_num == 0 and optimizer_idx == 0:
            self.epoch += 1

        if optimizer_idx == 0:
            batch = self(batch)
            reconstruction_loss = self.reconstruction.loss(batch)
            prior_loss = self.prior.loss(batch)

            batch['reconstruction_loss'] = reconstruction_loss / (self.objects['batch_size'] * self.objects['num_mel'] * self.objects['context'])
            batch['prior_loss'] = prior_loss / self.objects['batch_size']
            batch['loss'] = reconstruction_loss + prior_loss

            if batch_num == 0:
                self.logger_.log_reconstruction(batch, self.epoch)

            self.logger_.log_training_step(batch, self.step)
            self.step += 1
        else:
            raise AttributeError

        return {
            'loss': batch['loss'],
            'tqdm': {'loss': batch['loss']},
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

        self.logger_.log_vae_validation(outputs, self.step, self.epoch)
        return {
            'val_loss': np.concatenate([o['scores'].detach().cpu().numpy() for o in outputs]).mean()
        }

    def test_step(self, batch, batch_num):
        return self.validation_step(batch, batch_num)

    def test_end(self, outputs):
        # TODO: add new logging method
        # self.result = self.logger_.log_testing(outputs)
        self.logger_.close()
        return {}

    def train_dataloader(self):

        if self.objects['debug']:
            ds = torch.utils.data.Subset(self.objects['data_set'].get_whole_training_data_set(), np.arange(1024))
        else:
            ds = self.objects['data_set'].get_whole_training_data_set()

        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=self.objects['batch_size'],
            shuffle=True,
            num_workers=self.objects['num_workers'],
            drop_last=False
        )
        return dl


def configuration():
    seed = 1220
    deterministic = False
    id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
    log_path = os.path.join('..', 'experiment_logs', id)

    #####################
    # quick configuration, uses default parameters of more detailed configuration
    #####################

    machine_type = 0
    machine_id = 0

    latent_size = 40
    batch_size = 512

    debug = False
    if debug:
        epochs = 1
        num_workers = 0
    else:
        epochs = 50
        num_workers = 4

    learning_rate = 1e-4
    weight_decay = 0

    normalize = 'all'
    normalize_raw = True

    prior_class = 'priors.StandardNormalPrior'

    context = 11
    descriptor = "vae_training_{}_{}_{}_{}_{}_{}_{}_{}".format(
        prior_class,
        latent_size,
        batch_size,
        learning_rate,
        weight_decay,
        normalize,
        normalize_raw,
        context
    )

    ########################
    # detailed configuration
    ########################

    num_mel = 40
    n_fft = 512
    hop_size = 256

    prior = {
        'class': prior_class,
        'kwargs': {
            'latent_size': latent_size,
            'weight': 1
        }
    }

    data_set = {
        'class': 'data_sets.MCMDataSet',
        'kwargs': {
            'context': context,
            'num_mel': num_mel,
            'n_fft': n_fft,
            'hop_size': hop_size,
            'normalize': normalize,
            'normalize_raw': normalize_raw
        }
    }

    reconstruction = {
        'class': 'losses.MSE',
        'kwargs': {
            'weight': 1,
            'input_shape': '@data_set.observation_shape'
        }
    }

    auto_encoder_model = {
        'class': 'models.SamplingFCAE',
        'args': [
            '@data_set.observation_shape',
            '@reconstruction',
            '@prior'
        ]
    }

    lr_scheduler = {
        'class': 'torch.optim.lr_scheduler.StepLR',
        'args': [
            '@optimizer',
        ],
        'kwargs': {
            'step_size': epochs
        }
    }

    optimizer = {
        'class': 'torch.optim.Adam',
        'args': [
            '@auto_encoder_model.parameters()'
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


ex = Experiment('dcase2020_task2_vae_training')
cfg = ex.config(configuration)


@ex.automain
def run(_config, _run):
    experiment = VAEExperiment(_config, _run)
    return experiment.run()
