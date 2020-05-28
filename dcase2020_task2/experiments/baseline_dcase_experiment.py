from dcase2020_task2.experiments import BaseExperiment
from datetime import datetime
import os
import pytorch_lightning as pl
import torch
from sacred import Experiment
from dcase2020_task2.utils.logger import Logger
import torch.utils.data
# workaround...
from sacred import SETTINGS
SETTINGS['CAPTURE_MODE'] = 'sys'

class BaselineDCASEExperiment(BaseExperiment, pl.LightningModule):
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

    def training_step(self, batch_normal, batch_num, optimizer_idx=0):

        if batch_num == 0 and optimizer_idx == 0:
            self.epoch += 1

        if optimizer_idx == 0:
            batch_normal = self(batch_normal)
            reconstruction_loss = self.reconstruction.loss(batch_normal)
            prior_loss = self.prior.loss(batch_normal)

            batch_normal['reconstruction_loss'] = reconstruction_loss
            batch_normal['prior_loss'] = prior_loss
            batch_normal['loss'] = reconstruction_loss + prior_loss

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
        self.logger_.log_validation(outputs, self.step, self.epoch, all_ids=True)
        return {}

    def test_step(self, batch, batch_num):
        return self.validation_step(batch, batch_num)

    def test_end(self, outputs):
        self.result = self.logger_.log_testing(outputs, all_ids=True)
        self.logger_.close()
        return self.result

    def train_dataloader(self):
        dl = torch.utils.data.DataLoader(
            self.objects['data_set'].get_machine_training_data_set(self.machine_type),
            batch_size=self.objects['batch_size'],
            shuffle=True,
            num_workers=self.objects['num_workers'],
            drop_last=False
        )
        return dl

    def val_dataloader(self):
        dl = torch.utils.data.DataLoader(
            self.objects['data_set'].get_machine_validation_data_set(self.machine_type),
            batch_size=self.objects['batch_size'],
            shuffle=False,
            num_workers=self.objects['num_workers']
        )
        return dl

    def test_dataloader(self):
        dl = torch.utils.data.DataLoader(
            self.objects['data_set'].get_whole_validation_data_set(),
            batch_size=self.objects['batch_size'],
            shuffle=False,
            num_workers=self.objects['num_workers']
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

    latent_size = 8
    batch_size = 512

    debug = False
    if debug:
        epochs = 1
        num_workers = 0
    else:
        epochs = 100
        num_workers = 4

    learning_rate = 1e-3
    weight_decay = 0

    normalize = 'none'
    normalize_raw = False

    context = 5
    descriptor = "BaselineDCASEExperiment_{}_{}_{}_{}_{}_{}_{}".format(
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

    num_mel = 128
    n_fft = 1024
    hop_size = 512

    prior = {
        'class': 'dcase2020_task2.priors.NoPrior',
        'kwargs': {
            'latent_size': latent_size,
            'weight': 1.0
        }
    }

    data_set = {
        'class': 'dcase2020_task2.data_sets.MCMDataSet',
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
        'class': 'dcase2020_task2.losses.MSE',
        'kwargs': {
            'weight': 1.0,
            'input_shape': '@data_set.observation_shape'
        }
    }

    auto_encoder_model = {
        'class': 'dcase2020_task2.models.BaselineFCAE',
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
        'class': 'dcase2020_task2.trainers.PTLTrainer',
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


ex = Experiment('dcase2020_task2_BaselineDCASEExperiment')
cfg = ex.config(configuration)


@ex.automain
def run(_config, _run):
    experiment = BaselineDCASEExperiment(_config, _run)
    return experiment.run()
