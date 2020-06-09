from dcase2020_task2.experiments import BaseExperiment
import pytorch_lightning as pl
import torch
from sacred import Experiment
from dcase2020_task2.utils.logger import Logger
import os
import torch.utils.data
# workaround...
from sacred import SETTINGS

SETTINGS['CAPTURE_MODE'] = 'sys'
from datetime import datetime


class ClassificationExperiment(BaseExperiment, pl.LightningModule):

    def __init__(self, configuration_dict, _run):
        super().__init__(configuration_dict)

        # default stuff
        self.network = self.objects['model']
        self.loss = self.objects['loss']
        self.logger_ = Logger(_run, self, self.configuration_dict, self.objects)

        # experiment state variables
        self.epoch = -1
        self.step = 0
        self.result = None

        # will be set before each epoch
        self.normal_data_set = self.objects['data_set']
        self.abnormal_data_set = self.objects['abnormal_data_set']

        self.inf_data_loader = self.get_inf_data_loader(
            torch.utils.data.DataLoader(
                self.abnormal_data_set.training_data_set(),
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
        while True:
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
            abnormal_batch = next(self.inf_data_loader)

            normal_batch_size = len(batch_normal['observations'])
            abnormal_batch_size = len(abnormal_batch['observations'])

            device = batch_normal['observations'].device

            batch_normal['abnormal'] = torch.cat([
                torch.zeros(normal_batch_size, 1).to(device),
                torch.ones(abnormal_batch_size, 1).to(device)
            ])

            batch_normal['observations'] = torch.cat([
                batch_normal['observations'],
                abnormal_batch['observations']
            ])

            batch_normal = self(batch_normal)

            batch_normal = self.loss(batch_normal)

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
            'file_ids': batch['file_ids']
        }

    def validation_end(self, outputs):
        self.logger_.log_validation(outputs, self.step, self.epoch)
        return {}

    def test_step(self, batch, batch_num):
        return self.validation_step(batch, batch_num)

    def test_end(self, outputs):
        self.result = self.logger_.log_test(outputs)
        self.logger_.close()
        return self.result

    def train_dataloader(self):
        dl = torch.utils.data.DataLoader(
            self.objects['data_set'].training_data_set(),
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
    log_path = os.path.join('experiment_logs', id)

    #####################
    # quick configuration, uses default parameters of more detailed configuration
    #####################

    machine_type = 1
    machine_id = 0

    num_mel = 128
    n_fft = 1024
    hop_size = 512
    power = 2.0
    fmin = 0
    context = 128

    model_class = 'dcase2020_task2.models.CNN'
    hidden_size = 128
    num_hidden = 3
    dropout_probability = 0.0

    debug = False
    if debug:
        num_workers = 0
    else:
        num_workers = 4

    epochs = 100
    loss_class = 'dcase2020_task2.losses.AUC'
    batch_size = 512
    learning_rate = 1e-4
    weight_decay = 0

    same_type = True
    normalize_raw = True
    hop_all = False

    # TODO: change default descriptor
    descriptor = "ClassificationExperiment_Model:[{}_{}_{}_{}]_Training:[{}_{}_{}_{}]_Features:[{}_{}_{}_{}_{}_{}_{}]_Complement:[{}]{}".format(
        model_class,
        hidden_size,
        num_hidden,
        dropout_probability,
        loss_class,
        batch_size,
        learning_rate,
        weight_decay,
        normalize_raw,
        num_mel,
        context,
        n_fft,
        hop_size,
        power,
        fmin,
        same_type,
        seed
    )

    ########################
    # detailed configuration
    ########################

    data_set = {
        'class': 'dcase2020_task2.data_sets.MCMDataSet',
        'args': [
            machine_type,
            machine_id
        ],
        'kwargs': {
            'context': context,
            'num_mel': num_mel,
            'n_fft': n_fft,
            'hop_size': hop_size,
            'normalize_raw': normalize_raw,
            'power': power,
            'fmin': fmin,
            'hop_all': hop_all
        }
    }

    abnormal_data_set = {
        'class': 'dcase2020_task2.data_sets.ComplementMCMDataSet',
        'args': [
            machine_type,
            machine_id
        ],
        'kwargs': {
            'context': context,
            'num_mel': num_mel,
            'n_fft': n_fft,
            'hop_size': hop_size,
            'normalize_raw': normalize_raw,
            'power': power,
            'fmin': fmin,
            'hop_all': hop_all,
            'same_type': same_type
        }
    }

    loss = {
        'class': loss_class,
        'kwargs': {
            'weight': 1.0,
            'input_shape': '@data_set.observation_shape'
        }
    }

    model = {
        'class': model_class,
        'args': [
            '@data_set.observation_shape'
        ],
        'kwargs': {
            'hidden_size': hidden_size,
            'num_hidden': num_hidden,
            'dropout_probability': dropout_probability,
            'batch_norm': False
        }
    }

    lr_scheduler = {
        'class': 'torch.optim.lr_scheduler.StepLR',
        'args': [
            '@optimizer',
        ],
        'kwargs': {
            'step_size': 50,
            'gamma': 0.1
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


ex = Experiment('dcase2020_task2_ClassificationExperiment')
cfg = ex.config(configuration)


@ex.automain
def run(_config, _run):
    experiment = ClassificationExperiment(_config, _run)
    return experiment.run()
