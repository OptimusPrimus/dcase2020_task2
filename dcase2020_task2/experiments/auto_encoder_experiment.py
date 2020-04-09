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
import numpy as np
from datetime import datetime
import os
# workaround...
from sacred import SETTINGS
SETTINGS['CAPTURE_MODE'] = 'sys'
import sklearn.mixture


class AutoEncoderExperiment(pl.LightningModule, BaseExperiment):

    def __init__(self, configuration_dict, _run):
        super().__init__()

        self.configuration_dict = copy.deepcopy(configuration_dict)
        self.objects = create_objects_from_config(configuration_dict)

        if not os.path.exists(self.configuration_dict['log_path']):
            os.mkdir(self.configuration_dict['log_path'])

        if self.objects.get('deterministic'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.machine_type = self.objects['machine_type']
        self.machine_id = self.objects['machine_id']

        self.trainer = self.objects['trainer']
        self.model = self.objects['model']
        self.prior = self.objects['prior']
        self.data_set = self.objects['data_set']
        self.whole_data_set = self.data_set.get_whole_data_set()
        self.reconstruction = self.objects['reconstruction']

        self.machine_data_set = iter(
            self.get_data_loader(
                torch.utils.data.DataLoader(
                    self.data_set.validation_data_set(self.machine_type, self.machine_id),
                    batch_size=self.objects['batch_size'],
                    shuffle=True,
                    num_workers=self.objects['num_workers'],
                    drop_last=False
                )
            )
        )

        self.gmm = sklearn.mixture.GaussianMixture(
            n_components=16,
            covariance_type='diag',
            init_params='random',
            max_iter=1,
            warm_start=True,
            verbose=0
        )

        self.codes_buffer = []
        self.codes_buffer_length = 30

        self.logger_ = Logger(_run, self, self.configuration_dict, self.objects)
        self.epoch = -1
        self.step = 0
        self.result = None

    def get_data_loader(self, dl):
        device = "cuda:{}".format(self.trainer.root_gpu)
        for batch in iter(dl):
            for key in batch:
                if type(batch[key]) is torch.Tensor:
                    batch[key] = batch[key].to(device)
            yield batch

    def forward(self, batch):
        batch['epoch'] = self.epoch
        batch = self.model(batch)
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

            if len(self.codes_buffer) == self.codes_buffer_length:
                self.codes_buffer.pop(0)
            self.codes_buffer.append(batch_normal['codes'].detach().cpu().numpy())

            if self.step % self.codes_buffer_length == 0 and len(self.codes_buffer) == self.codes_buffer_length:
                self.gmm.fit(np.concatenate(self.codes_buffer))

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
        self.logger_.log_validation(outputs, self.step, self.epoch, gmm=self.gmm)
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
        self.result = self.logger_.log_testing(outputs, gmm=self.gmm)
        self.logger_.close()
        return self.result

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
            self.whole_data_set, # self.data_set.training_data_set(self.machine_type, self.machine_id),
            batch_size=self.objects['batch_size'],
            shuffle=True,
            num_workers=self.objects['num_workers'],
            drop_last=True
        )
        return dl

    def val_dataloader(self):
        dl = torch.utils.data.DataLoader(
            self.data_set.validation_data_set(self.machine_type, self.machine_id),
            batch_size=self.objects['batch_size'],
            shuffle=False,
            num_workers=self.objects['num_workers']
        )
        return dl

    def test_dataloader(self):
        dl = torch.utils.data.DataLoader(
            self.data_set.validation_data_set(self.machine_type, self.machine_id),
            batch_size=self.objects['batch_size'],
            shuffle=False,
            num_workers=self.objects['num_workers']
        )
        return dl

    def run(self):
        self.trainer.fit(self)
        self.trainer.test(self)
        return self.result


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

    epochs = 100
    num_workers = 0

    learning_rate = 1e-3
    weight_decay = 0

    normalize = 'all'
    normalize_raw = True


    ########################
    # detailed configuration
    ########################


    context = 5
    num_mel = 128
    n_fft = 1024
    hop_size = 512

    prior = {
        'class': 'priors.NoPrior',
        'kwargs': {
            'latent_size': latent_size,
            'weight': 1.0
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
            'weight': 1.0,
            'input_shape': '@data_set.observation_shape'
        }
    }

    model = {
        'class': 'models.BaselineFCAE',
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


ex = Experiment('dcase2020_task2_autoencoder')
cfg = ex.config(configuration)


@ex.automain
def run(_config, _run):
    experiment = AutoEncoderExperiment(_config, _run)
    return experiment.run()
