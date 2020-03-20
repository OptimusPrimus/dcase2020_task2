from torch.utils.tensorboard import SummaryWriter
import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import evaluation
import PIL

PIL.PILLOW_VERSION = PIL.__version__
import torchvision


class Logger:

    def __init__(self, _run, experiment_module, config, objects):
        self._run = _run
        self.experiment_module = experiment_module
        self.configuration_dict = config
        self.objects = objects

        self.log_dir = self.configuration_dict['log_path']
        self.testing_data_set = self.objects['testing_data_set']

        # self.writer = SummaryWriter(log_dir=self.log_dir)

        file = os.path.join(config['log_path'], 'conf.py')
        with open(file, 'wb') as config_dictionary_file:
            pickle.dump(config, config_dictionary_file)

    def log_training_step(self, batch, step):
        self.__log_metric__('training_loss', batch['loss'].item(), step)
        self.__log_metric__('training_prior_loss', batch['prior_loss'].item(), step)
        self.__log_metric__('training_reconstruction_loss', batch['reconstruction_loss'].item(), step)
        self.__log_metric__('training_auxiliary_loss', batch.get('auxiliary_loss', torch.tensor(0.0)).item(), step),
        self.__log_metric__('c', batch.get('c', 0.0), step)

    def log_validation(self, outputs, step, epoch):

        num_examples = 32
        factors = self.testing_data_set.sample_factors(num_examples, np.random.RandomState(seed=0))
        observations = self.testing_data_set.sample_observations(factors)

        batch = {
            'observations': observations.cuda(),
            'epoch': 0
        }

        batch = self.experiment_module(batch)

        images = torch.empty((64, *batch['observations'].shape[1:]))
        images[::2] = batch['observations']
        images[1::2] = batch['visualizations']
        image = torchvision.utils.make_grid(images, padding=5, pad_value=1)

        if epoch >= 0:
            self.__log_image__(
                image,
                'reconstruction_{:04d}.png'.format(epoch)
            )

    def log_testing(self, outputs):

        # save model
        self.__log_model__()

        # compute scores
        def representation_function(x):
            x = {
                'observations': x.cuda(),
                'epoch': 0
            }
            return self.experiment_module(x)['mus'].detach().cpu().numpy()

        beat_vae_score = float(evaluation.beta_vae(
            self.testing_data_set,
            representation_function,
            random_state=np.random.RandomState(0),
            batch_size=64,
            num_train=10_000,
            num_eval=10_000
        )['score'])

        factor_vae_score = float(evaluation.factor_vae(
            self.testing_data_set,
            representation_function,
            random_state=np.random.RandomState(0),
            batch_size=64,
            num_train=10_000,
            num_eval=5_000,
            num_variance_estimate=10_000,
            valid_dimension_threshold=0.05
        )['score'])

        mig_score = float(evaluation.mig(
            self.testing_data_set,
            representation_function,
            random_state=np.random.RandomState(0),
            num_train=10_000,
            batch_size=64
        )['score'])

        self.__log_metric__('beat_vae_score', beat_vae_score, 0)
        self.__log_metric__('factor_vae_score', factor_vae_score, 0)
        self.__log_metric__('mig_score', mig_score, 0)

        # create animation
        num_steps = 100
        num_samples = 1000
        factors = self.testing_data_set.sample_factors(num_samples, np.random.RandomState(seed=0))
        observations = self.testing_data_set.sample_observations(factors)
        batch = {
            'observations': observations.cuda(),
            'epoch': self.experiment_module.epoch
        }
        batch = self.experiment_module(batch)

        num_dims = batch['codes'].shape[1]
        num_rows = int(np.sqrt(num_dims))

        min = batch['codes'].min(dim=0)[0]
        max = batch['codes'].max(dim=0)[0]
        mean = batch['codes'].mean(dim=0)
        step_size = (max - min) / num_steps

        images = torch.empty((num_dims, num_steps, *batch['observations'][0].shape))

        for i in range(num_dims):
            for j in range(num_steps):
                mod = mean.clone().detach()
                mod[i] = min[i] + step_size[i] * j
                batch_ = {
                    'codes': mod.unsqueeze(0)
                }
                batch_ = self.experiment_module.auto_encoder_model.decode(batch_)
                images[i, j] = batch_['visualizations']

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xticks([])
        ax.set_yticks([])
        frames = []

        for j in range(num_steps):
            frames.append(
                torchvision.utils.make_grid(images[:, j], padding=5, pad_value=1, nrow=num_rows, normalize=True)
            )

        image = ax.imshow(frames[0].transpose(0, -1))

        def init():
            image.set_data(frames[0].transpose(0, -1))
            return image,

        def animate(i):
            image.set_data(frames[i].transpose(0, -1))
            return image,

        self.__log_video__(
            FuncAnimation(fig, animate, init_func=init, frames=num_steps, interval=100, blit=True),
            'animation.mp4'
        )

        return {
            'beta_vae_score': beat_vae_score,
            'factor_vae_score': factor_vae_score,
            'mig_score': mig_score
        }

    def __log_metric__(self, name, value, step):
        self._run.log_scalar(name, value, step)

    def __log_image__(self, image, file_name):
        torchvision.utils.save_image(
            image,
            os.path.join(
                self.configuration_dict['log_path'],
                file_name
            )
        )

    def __log_video__(self, animation, file_name):
        Writer = matplotlib.animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='PP'), bitrate=1800)
        path = os.path.join(self.configuration_dict['log_path'], file_name)
        animation.save(path, writer=writer)

    def __log_audio__(self):
        raise NotImplementedError

    def __log_model__(self):
        torch.save({
            'state_dict': self.experiment_module.state_dict(),
        },
            os.path.join(self.configuration_dict['log_path'], 'model.ckpt')
        )

    def close(self):
        # self.writer.close()
        pass
