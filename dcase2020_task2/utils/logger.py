from torch.utils.tensorboard import SummaryWriter
import os
import pickle
import torch
import numpy as np
import matplotlib
from matplotlib.animation import FuncAnimation
import PIL
from sklearn import metrics
from dcase2020_task2.data_sets.mcm_dataset import TRAINING_ID_MAP, INVERSE_CLASS_MAP, CLASS_MAP
import matplotlib.pyplot as plt
import pandas as pd

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
        file = os.path.join(config['log_path'], 'conf.pl')
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

    def log_validation(self, outputs, step, epoch):

        if epoch == -1:
            return dict()

        scores_mean, scores_max, _, ground_truth, file_ids, machine_types, machine_ids = self.__batches_to_per_file_scores__(
            outputs,
            aggregation_fun=None
        )

        result = self.__compute_metrics__(scores_mean, scores_max, ground_truth, machine_types, machine_ids)

        # log validation results
        for machine_type in result:
            for machine_id in result[machine_type]:

                self.__log_metric__(
                    f'{machine_type}_{machine_id}_validation_auroc_mean',
                    result[machine_type][machine_id][0],
                    step
                )
                self.__log_metric__(
                    f'{machine_type}_{machine_id}_validation_pauroc_mean',
                    result[machine_type][machine_id][1],
                    step
                )
                self.__log_metric__(
                    f'{machine_type}_{machine_id}_validation_auroc_max',
                    result[machine_type][machine_id][2],
                    step
                )
                self.__log_metric__(
                    f'{machine_type}_{machine_id}_validation_pauroc_max',
                    result[machine_type][machine_id][3],
                    step
                )

        return result

    def log_test(self, outputs):

        scores_mean, scores_max, _, ground_truth, file_ids, machine_types, machine_ids = self.__batches_to_per_file_scores__(
            outputs,
            aggregation_fun=None
        )

        result = self.__compute_metrics__(scores_mean, scores_max, ground_truth, machine_types, machine_ids)
        file_ids = np.array([f.split(os.sep)[-1] for f in file_ids])

        self.__plot_score_distribution__(scores_mean, scores_max, ground_truth, machine_types, machine_ids)

        # save predictions to file predictions ...
        for machine_type in result:
            # type back to int ...
            machine_type = CLASS_MAP[machine_type]

            for machine_id in result[INVERSE_CLASS_MAP[machine_type]]:

                # save predictions ...
                indices = np.where(np.logical_and(machine_types == machine_type, machine_ids == machine_id))[0]

                path_mean = os.path.join(
                    self.log_dir,
                    f'anomaly_score_{INVERSE_CLASS_MAP[machine_type]}_id_{machine_id}_mean.csv'
                )

                path_max = os.path.join(
                    self.log_dir,
                    f'anomaly_score_{INVERSE_CLASS_MAP[machine_type]}_id_{machine_id}_max.csv'
                )

                pd.DataFrame(
                    list(zip(file_ids[indices], scores_mean[indices]))
                ).to_csv(path_mean, index=False, header=False)

                pd.DataFrame(
                    list(zip(file_ids[indices], scores_max[indices]))
                ).to_csv(path_max, index=False, header=False)

        return result

    def log_image_reconstruction(self, batch, epoch, num_images=10):

        num_images = np.minimum(len(batch), num_images)

        assert num_images > 0
        assert batch.get('observations')
        assert batch.get('visualizations')

        grid_img = torchvision.utils.make_grid(
            torch.cat(
                batch['observations'][:num_images],
                batch['visualizations'][:num_images]
            ),
            nrow=num_images
        )

        self.__log_image__(grid_img, f'{epoch}_reconstruction_x.png')

    def __plot_score_distribution__(self, scores_mean, scores_max, ground_truth, machine_types, machine_ids):

        # select samples with matching machine_types and machine ids only
        unique_machine_types = np.unique(machine_types)
        unique_machine_ids = [np.unique(machine_ids[machine_types == r]) for r in unique_machine_types]

        n_rows = len(unique_machine_types)
        n_cols = np.max([len(c) for c in unique_machine_ids])

        # print distribution
        plt.ioff()
        plt.figure(figsize=(5 * n_cols, 5 * n_rows))

        bins = np.linspace(scores_mean.min(), scores_mean.max(), 50)

        for i, typ in enumerate(unique_machine_types):
            for j, id in enumerate(unique_machine_ids[i]):

                plt.subplot(n_rows, n_cols, ((i * n_cols) + j) + 1)

                x_normal = scores_mean[
                    np.logical_and(ground_truth == 0, np.logical_and(machine_ids == id, machine_types == typ))
                ]

                x_abnormal = scores_mean[
                    np.logical_and(ground_truth == 1, np.logical_and(machine_ids == id, machine_types == typ))
                ]

                plt.hist(x_normal, bins, alpha=0.5, label='normal')
                plt.hist(x_abnormal, bins, alpha=0.5, label='abnormal')

                if i == 0 and j == 0:
                    plt.legend(loc='upper right')

        plt.savefig(os.path.join(self.log_dir, 'score_distribution.png'), bbox_inches='tight')
        plt.close()

    def __compute_metrics__(
            self,
            scores_mean,
            scores_max,
            ground_truth,
            machine_types,
            machine_ids
    ):

        # select samples with matching machine_types and machine ids only
        unique_machine_types = np.unique(machine_types)
        unique_machine_ids = [np.unique(machine_ids[machine_types == r]) for r in unique_machine_types]

        # put results into dictionary
        result = dict()
        for i, machine_type in enumerate(unique_machine_types):
            machine_type = INVERSE_CLASS_MAP[machine_type]
            for machine_id in unique_machine_ids[i]:
                result.setdefault(machine_type, dict())[int(machine_id)] = self.__rauc_from_score__(
                    scores_mean,
                    scores_max,
                    ground_truth,
                    machine_types,
                    machine_ids,
                    CLASS_MAP[machine_type],
                    machine_id
                )

        return result

    @staticmethod
    def __rauc_from_score__(scores_mean, scores_max, ground_truth, machine_types, machine_ids, machine_type, id,
                            max_fpr=0.1):
        ground_truth_ = ground_truth[np.logical_and(machine_types == machine_type, machine_ids == id)]
        scores_mean_ = scores_mean[np.logical_and(machine_types == machine_type, machine_ids == id)]
        scores_max_ = scores_max[np.logical_and(machine_types == machine_type, machine_ids == id)]

        return float(metrics.roc_auc_score(ground_truth_, scores_mean_)), \
               float(metrics.roc_auc_score(ground_truth_, scores_mean_, max_fpr=max_fpr)), \
               float(metrics.roc_auc_score(ground_truth_, scores_max_)), \
               float(metrics.roc_auc_score(ground_truth_, scores_max_, max_fpr=max_fpr))

    @staticmethod
    def __batches_to_per_file_scores__(outputs, aggregation_fun=None):

        # extract targets, scores and file_ids fom batches
        targets_ = np.concatenate([o['targets'].detach().cpu().numpy() for o in outputs])
        predictions_ = np.concatenate([o['scores'].detach().cpu().numpy() for o in outputs])
        machine_types_ = np.concatenate([o['machine_types'].detach().cpu().numpy() for o in outputs])
        machine_ids_ = np.concatenate([o['machine_ids'].detach().cpu().numpy() for o in outputs])

        file_indices_map = {}
        idx = 0
        for o in outputs:
            for file_name in o['file_ids']:
                file_indices_map.setdefault(file_name, []).append(idx)
                idx += 1

        targets = []
        scores_mean = []
        scores_max = []
        scores_custom = []
        machine_types = []
        machine_ids = []
        unique_files = np.unique(list(file_indices_map.keys()))

        for f in unique_files:
            indices = file_indices_map[f]

            targets.append(targets_[indices[0]])
            scores_mean.append(np.mean(predictions_[indices]))
            scores_max.append(np.max(predictions_[indices]))
            if aggregation_fun:
                scores_custom.append(aggregation_fun(predictions_[indices]))
            machine_types.append(machine_types_[indices[0]])
            machine_ids.append(machine_ids_[indices[0]])

            assert all(machine_types[-1] == machine_types_[indices])
            assert all(machine_ids[-1] == machine_ids_[indices])
            assert all(targets[-1] == targets_[indices])


        return np.array(scores_mean), \
        np.array(scores_max), \
        np.array(scores_custom), \
        np.array(targets), \
        np.array(unique_files), \
        np.array(machine_types), \
        np.array(machine_ids)

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
