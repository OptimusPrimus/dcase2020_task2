from torch.utils.tensorboard import SummaryWriter
import os
import pickle
import torch
import numpy as np
import matplotlib
from matplotlib.animation import FuncAnimation
import PIL
from sklearn import metrics
from dcase2020_task2.data_sets.mcm_dataset import TRAINING_ID_MAP
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

        self.file_indices = dict()

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

    def log_reconstruction(self, batch, epoch):
        for i, (observation, reconstructed) in enumerate(zip(batch['observations'], batch['predictions'])):
            if i == 10:
                break
            self.__log_image__(observation, '{}_{}_image_x.png'.format(epoch, i))
            self.__log_image__(reconstructed, '{}_{}_image_xhat.png'.format(epoch, i))


    def log_validation(self, outputs, step, epoch, all_ids=False):

        if epoch == -1:
            return None

        scores_mean, _, _, _, _ = self.__batches_to_per_file_scores__(outputs, aggregation_fun=np.mean)
        scores_max, ground_truth, file_id, machine_types, machine_ids = self.__batches_to_per_file_scores__(outputs,
                                                                                                            aggregation_fun=np.max)

        # select samples with matching machine_types and machine ids only
        plt.ioff()
        plt.figure(figsize=(24, 20))

        bins = np.linspace(scores_mean.min(), scores_mean.max(), 50)

        for i, typ in enumerate(np.arange(6)):
            for j, id in enumerate(TRAINING_ID_MAP[typ]):
                plt.subplot(6, 4, ((i * 4) + j) + 1)

                x_normal = scores_mean[
                    np.logical_and(ground_truth == 0, np.logical_and(machine_ids == id, machine_types == typ))]
                x_abnormal = scores_mean[
                    np.logical_and(ground_truth == 1, np.logical_and(machine_ids == id, machine_types == typ))]

                plt.hist(x_normal, bins, alpha=0.5, label='normal')
                plt.hist(x_abnormal, bins, alpha=0.5, label='abnormal')

                if i == 0 and j == 0:
                    plt.legend(loc='upper right')

        plt.savefig(os.path.join(self.log_dir, 'score_distribution_{}.png'.format(epoch)), bbox_inches='tight')
        plt.close()

        auroc_mean = []
        pauroc_mean = []
        auroc_max = []
        pauroc_max = []

        for id in (TRAINING_ID_MAP[self.machine_type] if all_ids else [self.machine_id]):
            ground_truth_ = ground_truth[np.logical_and(machine_types == self.machine_type, machine_ids == id)]
            scores_mean_ = scores_mean[np.logical_and(machine_types == self.machine_type, machine_ids == id)]
            scores_max_ = scores_max[np.logical_and(machine_types == self.machine_type, machine_ids == id)]

            auroc_mean.append(metrics.roc_auc_score(ground_truth_, scores_mean_))
            pauroc_mean.append(metrics.roc_auc_score(ground_truth_, scores_mean_, max_fpr=0.1))
            auroc_max.append(metrics.roc_auc_score(ground_truth_, scores_max_))
            pauroc_max.append(metrics.roc_auc_score(ground_truth_, scores_max_, max_fpr=0.1))

            if epoch != -2:
                self.__log_metric__('validation_auroc_mean_{}'.format(id if all_ids else ""), auroc_mean[-1], step)
                self.__log_metric__('validation_pauroc_mean_{}'.format(id if all_ids else ""), pauroc_mean[-1], step)
                self.__log_metric__('validation_auroc_max_{}'.format(id if all_ids else ""), auroc_max[-1], step)
                self.__log_metric__('validation_pauroc_max_{}'.format(id if all_ids else ""), pauroc_max[-1], step)

        return {
            'auroc_mean': float(np.mean(auroc_mean)),
            'pauroc_mean': float(np.mean(pauroc_mean)),
            'auroc_max': float(np.mean(auroc_max)),
            'pauroc_max': float(np.mean(pauroc_max))
        }

    def log_vae_validation(self, outputs, step, epoch):

        if epoch == -1:
            return None

        errors = np.concatenate([o['scores'].detach().cpu().numpy() for o in outputs])
        machine_types = np.concatenate([o['machine_types'].detach().cpu().numpy() for o in outputs])
        machine_ids = np.concatenate([o['machine_ids'].detach().cpu().numpy() for o in outputs])

        for ty in range(6):
            for id in TRAINING_ID_MAP[ty]:
                idxs = np.logical_and(machine_types == ty, machine_ids == id)
                if np.any(idxs):
                    self.__log_metric__(
                        'validation_reconstruction_error_{}_{}'.format(id, ty),
                        np.mean(errors[idxs]),
                        step
                    )

        self.__log_metric__('validation_reconstruction_error', np.mean(errors), step)

    def log_testing(self, outputs, all_ids=False):
        return self.log_validation(outputs, 0, -2, all_ids=all_ids)


    def __batches_to_per_file_scores__(self, outputs, aggregation_fun=np.mean):

        # extract targets, scores and file_ids fom batches
        targets = np.concatenate([o['targets'].detach().cpu().numpy() for o in outputs])
        predictions = np.concatenate([o['scores'].detach().cpu().numpy() for o in outputs])
        file_ids = np.concatenate([o['file_ids'].detach().cpu().numpy() for o in outputs])
        machine_types = np.concatenate([o['machine_types'].detach().cpu().numpy() for o in outputs])
        machine_ids = np.concatenate([o['machine_ids'].detach().cpu().numpy() for o in outputs])

        unique_files = np.unique(file_ids)

        ground_truth = np.array([targets[file_ids == f][0] for f in unique_files])
        scores = np.array([aggregation_fun(predictions[file_ids == f]) for f in unique_files])
        machine_types = np.array([machine_types[file_ids == f] for f in unique_files])
        machine_id = np.array([machine_ids[file_ids == f] for f in unique_files])

        return scores, ground_truth, unique_files, machine_types, machine_id


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
