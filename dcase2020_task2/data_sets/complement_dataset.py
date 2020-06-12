import os
import torch.utils.data
from dcase2020_task2.data_sets import BaseDataSet, CLASS_MAP, INVERSE_CLASS_MAP, TRAINING_ID_MAP, ALL_ID_MAP
from dcase2020_task2.data_sets import MachineDataSet
import numpy as np

valid_types = {
    0: [1, 2, 5],
    1: [0, 2, 5],
    2: [0, 1, 5],
    5: [0, 1, 2],
    3: [4],
    4: [3],
}


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
            hop_all=False
    ):

        assert type(machine_type) == int and type(machine_id) == int

        self.data_root = data_root
        self.context = context
        self.num_mel = num_mel
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.power = power
        self.fmin = fmin
        self.hop_all = hop_all
        self.normalize_raw = normalize_raw

        kwargs = {
            'data_root': self.data_root,
            'context': self.context,
            'num_mel': self.num_mel,
            'n_fft': self.n_fft,
            'hop_size': self.hop_size,
            'power': self.power,
            'normalize': self.normalize_raw,
            'fmin': self.fmin,
            'hop_all': self.hop_all
        }

        training_sets = []

        data = []
        for type_ in ALL_ID_MAP:
            for id_ in ALL_ID_MAP[type_]:
                if type_ != machine_type or (id_ != machine_id and machine_id != -1):
                    t = MachineDataSet(type_, id_, mode='training', **kwargs)
                    data.append(t.data)
                    training_sets.append(t)
        data = np.concatenate(data, axis=-1)

        self.mean = data.mean(axis=1, keepdims=True)
        self.std = data.std(axis=1, keepdims=True)
        del data

        self.training_set = torch.utils.data.ConcatDataset(training_sets)

    @property
    def observation_shape(self) -> tuple:
        return 1, self.num_mel, self.context

    def training_data_set(self):
        return self.training_set

    def validation_data_set(self):
        raise NotImplementedError

    def mean_std(self):
        return self.mean, self.std

