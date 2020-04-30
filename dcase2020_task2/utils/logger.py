from torch.utils.tensorboard import SummaryWriter
import os
import pickle
import torch
import numpy as np
import matplotlib
from matplotlib.animation import FuncAnimation
import PIL
from sklearn import metrics
import matplotlib.pyplot as plt

PIL.PILLOW_VERSION = PIL.__version__
import torchvision


class Logger:

    def __init__(self, _run, experiment_module, config, objects):
        self._run = _run
        self.experiment_module = experiment_module
        self.configuration_dict = config
        self.objects = objects

        self.log_dir = self.configuration_dict['log_path']

        self.machine_type = self.objects['machine_type']
        self.machine_id = self.objects['machine_id']

        # self.writer = SummaryWriter(log_dir=self.log_dir)
        file = os.path.join(config['log_path'], 'conf.py')
        with open(file, 'wb') as config_dictionary_file:
            pickle.dump(config, config_dictionary_file)

    def log_training_step(self, batch, step):
        if step % 100 == 0:
            for key in batch:
                if type(batch[key]) == np.ndarray and batch[key].dims == 1 and batch[key].shape[0]:
                    self.__log_metric__(key, float(batch[key]), step)
                elif type(batch[key]) in [float, int]:
                    self.__log_metric__(key, float(batch[key]), step)
                elif type(batch[key]) == torch.Tensor and batch[key].ndim == 0:
                    self.__log_metric__(key, batch[key].item(), step)
                elif type(batch[key]) == torch.Tensor and batch[key].ndim == 1 and batch[key].shape[0] == 1:
                    self.__log_metric__(key, batch[key].item(), step)

    def log_validation(self, outputs, step, epoch):

        if epoch == -1:
            return None

        scores_mean, _, _, _, _ = self.__batches_to_per_file_scores__(outputs, aggregation_fun=np.mean)
        scores_max, ground_truth, file_id, machine_types, machine_ids = self.__batches_to_per_file_scores__(outputs, aggregation_fun=np.max)

        # select samples with matching machine_types and machine ids only
        ground_truth = ground_truth[np.logical_and(machine_types == self.machine_type, machine_ids == self.machine_id)]
        scores_mean = scores_mean[np.logical_and(machine_types == self.machine_type, machine_ids == self.machine_id)]
        scores_max = scores_max[np.logical_and(machine_types == self.machine_type, machine_ids == self.machine_id)]

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
            'auroc_mean': float(auroc_mean),
            'pauroc_mean': float(pauroc_mean),
            'auroc_max': float(auroc_max),
            'pauroc_max': float(pauroc_max),
        }

    def log_testing(self, outputs):
        return self.log_validation(outputs, 0, -2)

    def __batches_to_per_file_scores__(self, outputs, aggregation_fun=np.mean):

        # extract targets, scores and file_ids fom batches
        targets = np.concatenate([o['targets'].detach().cpu().numpy() for o in outputs])
        predictions = np.concatenate([o['scores'].detach().cpu().numpy() for o in outputs])
        file_ids = np.concatenate([o['file_ids'].detach().cpu().numpy() for o in outputs])
        machine_types = np.concatenate([o['machine_types'].detach().cpu().numpy() for o in outputs])
        machine_ids = np.concatenate([o['machine_ids'].detach().cpu().numpy() for o in outputs])

        # compute per file aggregated targets and scores
        files = np.unique(np.stack([targets, file_ids, machine_types, machine_ids], axis=1), axis=0)

        scores = np.array([
            aggregation_fun(
                predictions[
                    np.logical_and(targets == target,
                        np.logical_and(file_ids == file_id,
                                   np.logical_and(machine_types == mty, machine_ids == mid)))
                ]
            )
            for target, file_id, mty, mid in files
        ])

        return scores, files[:, 0], files[:, 1], files[:, 2], files[:, 3]

    def __log_metric__(self, name, value, step):

        if value is None:
            value = 0.0

        if type(value) == torch.Tensor:
            value = value.item()
        elif type(value) == np.ndarray:
            value = float(value)

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
