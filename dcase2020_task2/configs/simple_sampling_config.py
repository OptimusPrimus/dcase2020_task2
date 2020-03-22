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
    prior_class = 'priors.NoPrior'
    latent_size = 40

    machine_type = 0
    batch_size = 512

    epochs = 100
    num_workers = 4

    ########################
    # detailed configuration
    ########################

    # set default values for different priors
    if prior_class == 'priors.NoPrior':
        prior = {
            'class': prior_class,
            'kwargs': {
                'latent_size': latent_size,
                'weight': 1.0
            }
        }
    elif prior_class == 'priors.StandardNormalPrior':
        prior = {
            'class': prior_class,
            'kwargs': {
                'latent_size': latent_size,
                'weight': 1.0,
                'c_max': 0.0,
                'c_stop_epoch': epochs
            }
        }
    elif prior_class == 'priors.SimplexPrior' or prior_class == 'priors.OrthogonalPrior':
        prior = {
            'class': prior_class,
            'kwargs': {
                'min_anneal': 0.0,
                'max_anneal': 1.0,
                'anneal_stop_epoch': epochs,
                'latent_size': latent_size,
                'weight': 1.0,
                'c_max': 0.0,
                'c_stop_epoch': epochs
            }
        }
    elif prior_class == 'priors.BetaTCVaePrior':
        # TODO: add default parameters
        prior = {
            'class': prior_class,
            'args': [
                '@training_data_set.size'
            ],
            'kwargs': {
                'min_anneal': 0.0,
                'max_anneal': 1.0,
                'anneal_stop_epoch': epochs,
                'latent_size': latent_size,
                'weight': 1.0,
                'c_max': 0.0,
                'c_stop_epoch': epochs
            }
        }
    elif prior_class == 'priors.DIPVaePrior':
        # TODO: add default parameters
        prior = {
            'class': prior_class,
            'args': [
                '@training_data_set.size'
            ],
            'kwargs': {
                'min_anneal': 0.0,
                'max_anneal': 1.0,
                'anneal_stop_epoch': epochs,
                'latent_size': latent_size,
                'weight': 1.0,
                'c_max': 0.0,
                'c_stop_epoch': epochs,
                'dip_first': False,
            }
        }

    training_data_set = {
        'class': 'data_sets.MCMDataset',
        'kwargs': {
            'mode': 'training',
            'machine_type': machine_type,
            'context': 11,
            'num_mel': 40,
            'n_fft': 512,
            'hop_size': 256
        }
    }

    validation_data_set = {
        'class': 'data_sets.MCMDataset',
        'kwargs': {
            'mode': 'validation',
            'machine_type': machine_type,
            'context': 11,
            'num_mel': 40,
            'n_fft': 512,
            'hop_size': 256
        }
    }

    reconstruction = {
        'class': 'reconstructions.MSE',
        'kwargs': {
            'weight': 1.0,
            'input_shape': '@training_data_set.observation_shape'
        }
    }

    auto_encoder_model = {
        'class': 'models.FCBaseLine',
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
            'lr': 1e-3,
            'betas': (0.9, 0.999),
            'amsgrad': False,
            'weight_decay': 1e-4,
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
        }
    }
