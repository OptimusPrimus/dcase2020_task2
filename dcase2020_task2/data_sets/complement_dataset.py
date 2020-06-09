import os
import torch.utils.data
from dcase2020_task2.data_sets import BaseDataSet, CLASS_MAP, INVERSE_CLASS_MAP, TRAINING_ID_MAP, ALL_ID_MAP
from dcase2020_task2.data_sets import MachineDataSet


class ComplementMCMDataSet(BaseDataSet):

    def __init__(
            self,
            machine_type,
            machine_id,
            data_root=os.path.join(os.path.expanduser('~'), 'shared', 'dcase2020_task2'),
            context=5,
            num_mel=128,
            n_fft=1024,
            hop_size=512,
            power=1.0,
            fmin=0,
            normalize_raw=False,
            normalize=None,
            hop_all=False,
            same_type=False
    ):
        self.data_root = data_root
        self.context = context
        self.num_mel = num_mel
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.power = power
        self.fmin = fmin
        self.normalize = normalize
        self.hop_all = hop_all

        assert type(machine_type) == int and type(machine_id) == int

        kwargs = {
            'data_root': self.data_root,
            'context': self.context,
            'num_mel': self.num_mel,
            'n_fft': self.n_fft,
            'hop_size': self.hop_size,
            'power': power,
            'normalize': normalize_raw,
            'fmin': fmin,
            'hop_all': hop_all
        }

        training_set = MachineDataSet(machine_type, machine_id, mode='training', **kwargs)
        validation_set = MachineDataSet(machine_type, machine_id, mode='validation', **kwargs)

        if normalize is None:
            mean = training_set.data.mean(axis=1, keepdims=True)
            std = training_set.data.std(axis=1, keepdims=True)
            training_set.data = (training_set.data - mean) / std
            validation_set.data = (validation_set.data - mean) / std
        else:
            assert type(normalize) == tuple
            assert len(normalize) == 2
            mean, std = normalize
            training_set.data = (training_set.data - mean) / std
            validation_set.data = (validation_set.data - mean) / std

        training_sets = []
        # validation_sets = []

        for type_ in ALL_ID_MAP:
            for id_ in ALL_ID_MAP[type_]:
                if type_ != machine_type or (id_ != machine_id and same_type):
                    t = MachineDataSet(type_, id_, mode='training', **kwargs)
                    t.data = (t.data - mean) / std
                    training_sets.append(t)

                    # don't load validation set ...
                    # v = MachineDataSet(type_, id_, mode='validation', **kwargs)
                    # v.data = (v.data - mean) / std
                    # validation_sets.append(v)

        self.training_set = torch.utils.data.ConcatDataset(training_sets)
        # self.validation_set = torch.utils.data.ConcatDataset(validation_sets)
        self.mean = mean
        self.std = std

    @property
    def observation_shape(self) -> tuple:
        return 1, self.num_mel, self.context

    def training_data_set(self):
        return self.training_set

    def validation_data_set(self):
        raise NotImplementedError

    def mean_std(self):
        return self.mean, self.std

