from datetime import datetime
import os


def configuration():
    seed = 1220
    deterministic = False
    id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_path = os.path.join('..', 'experiment_logs', id)

    #####################
    # quick configuration, uses default parameters of more detailed configuration
    #####################
    latent_size = 40

    machine_type = 0
    batch_size = 512

    epochs = 500
    num_workers = 4

    learning_rate = 1e-4
    weight_decay = 1e-5

    rho = 0.2

    context = 11
    num_mel = 40
    n_fft = 512
    hop_size = 256

    ########################
    # detailed configuration
    ########################

    prior = {
        'class': 'priors.NoPrior',
        'kwargs': {
            'latent_size': latent_size,
            'weight': 1.0
        }
    }


    training_data_set = {
        'class': 'data_sets.MCMDataset',
        'kwargs': {
            'mode': 'training',
            'machine_type': machine_type,
            'context': context,
            'num_mel': num_mel,
            'n_fft': n_fft,
            'hop_size': hop_size
        }
    }

    validation_data_set = {
        'class': 'data_sets.MCMDataset',
        'kwargs': {
            'mode': 'validation',
            'machine_type': machine_type,
            'context': context,
            'num_mel': num_mel,
            'n_fft': n_fft,
            'hop_size': hop_size
        }
    }

    reconstruction = {
        'class': 'reconstructions.NP',
        'kwargs': {
            'weight': 1.0,
            'input_shape': '@training_data_set.observation_shape',
            'rho': rho
        }
    }

    auto_encoder_model = {
        'class': 'models.SamplingFCAE',
        'args': [
            '@training_data_set.observation_shape',
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

    #### Generator

    generator_prior = {
        'class': 'priors.SimpleKLPrior',
        'kwargs': {
            'latent_size': latent_size,
            'weight': 1.0,
            'c_max': 0.0,
            'c_stop_epoch': epochs
        }
    }

    generator_reconstruction = {
        'class': 'reconstructions.MSE',
        'kwargs': {
            'weight': 1.0,
            'input_shape': '@training_data_set.observation_shape'
        }
    }

    generator_model = {
        'class': 'models.SamplingFCGenerator',
        'args': [
            '@training_data_set.observation_shape',
            '@generator_reconstruction',
            '@generator_prior'
        ],
        'kwargs': {
            'encoder': '@auto_encoder_model.encoder'
        }
   }

    generator_lr_scheduler = {
        'class': 'torch.optim.lr_scheduler.StepLR',
        'args': [
            '@generator_optimizer',
        ],
        'kwargs': {
            'step_size': epochs
        }
    }

    generator_optimizer = {
        'class': 'torch.optim.Adam',
        'args': [
            '@generator_model.parameters()'
        ],
        'kwargs': {
            'lr': learning_rate,
            'betas': (0.9, 0.999),
            'amsgrad': False,
            'weight_decay': weight_decay,
        }
    }

    ### trainer

    trainer = {
        'class': 'trainers.PTLTrainer',
        'kwargs': {
            'max_epochs': epochs,
            'checkpoint_callback': False,
            'logger': False,
            'early_stop_callback': False,
            'gpus': [0],
        }
    }