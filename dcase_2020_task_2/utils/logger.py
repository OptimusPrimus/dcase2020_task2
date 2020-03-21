from torch.utils.tensorboard import SummaryWriter
import os
import pickle
import torch
import numpy as np
import matplotlib
from matplotlib.animation import FuncAnimation
import PIL
from sklearn import metrics

PIL.PILLOW_VERSION = PIL.__version__
import torchvision


class Logger:

    def __init__(self, _run, experiment_module, config, objects):
        self._run = _run
        self.experiment_module = experiment_module
        self.configuration_dict = config
        self.objects = objects

        self.log_dir = self.configuration_dict['log_path']
        self.testing_data_set = self.objects['validation_data_set']

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

        if epoch == -1:
            return None

        targets = []
        predictions = []
        file_ids = []
        for o in outputs:
            targets.append(o['targets'].detach().cpu().numpy())
            predictions.append(o['predictions'].detach().cpu().numpy())
            file_ids.append(o['file_ids'].detach().cpu().numpy())

        targets = np.concatenate(targets)
        predictions = np.concatenate(predictions)
        file_ids = np.concatenate(file_ids)

        ground_truth = []
        scores_mean = []
        scores_max = []

        for file_id in np.unique(file_ids):
            selected = file_ids == file_id
            ground_truth.append(targets[selected][0])
            scores_mean.append(predictions[selected].mean())
            scores_max.append(predictions[selected].max())

        ground_truth = np.array(ground_truth)
        scores_mean = np.array(scores_mean)
        scores_max = np.array(scores_max)

        auroc_mean = metrics.roc_auc_score(ground_truth, scores_mean)
        pauroc_mean = metrics.roc_auc_score(ground_truth, scores_mean, max_fpr=0.1)

        auroc_max = metrics.roc_auc_score(ground_truth, scores_max)
        pauroc_max = metrics.roc_auc_score(ground_truth, scores_max, max_fpr=0.1)

        if epoch != -2:
            self.__log_metric__('validation_auroc_mean', auroc_mean, step)
            self.__log_metric__('validation_pauroc_mean', pauroc_mean, step)

            self.__log_metric__('validation_auroc_max', auroc_max, step)
            self.__log_metric__('validation_pauroc_max', pauroc_max, step)

        return {
            'auroc_mean': auroc_mean,
            'pauroc_mean': pauroc_mean
        }

    def log_testing(self, outputs):
        return self.log_validation(outputs, 0, -2)

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

    @staticmethod
    def auroc(normal_scores, abnormal_scores):
        if len(normal_scores) == 0 or len(abnormal_scores) == 0:
            return 0
        return np.mean(np.where((abnormal_scores[None, :] - normal_scores[:, None]) > 0, 1, 0))

    @staticmethod
    def pauroc(normal_scores, abnormal_scores, p=0.1):
        if len(normal_scores) == 0 or len(abnormal_scores) == 0:
            return 0
        num_samp = int(len(normal_scores) * p)
        normal_scores = np.sort(normal_scores)[:num_samp]
        return np.mean(np.where((abnormal_scores[:, None] - normal_scores[None, :]) > 0, 1, 0))

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
