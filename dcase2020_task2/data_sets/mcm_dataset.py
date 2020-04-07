import os
import torch.utils.data
import glob
import data_sets
import librosa
import sys
import numpy as np
import random

CLASS_MAP = {
    'fan': 0,
    'pump': 1,
    'slider': 2,
    'ToyCar': 3,
    'ToyConveyor': 4,
    'valve': 5
}

TRAINING_ID_MAP = {
    0: [0, 2, 4, 6],
    1: [0, 2, 4, 6],
    2: [0, 2, 4, 6],
    3: [1, 2, 3, 4],
    4: [1, 2, 3],
    5: [0, 2, 4, 6]
}

INVERSE_CLASS_MAP = {
    0: 'fan',
    1: 'pump',
    2: 'slider',
    3: 'ToyCar',
    4: 'ToyConveyor',
    5: 'valve'
}


class MCMDataSet(data_sets.BaseDataSet):

    def __init__(
            self,
            data_root=os.path.join(os.path.expanduser('~'), 'shared', 'dcase2020_task2'),
            context=5,
            num_mel=128,
            n_fft=1024,
            hop_size=512,
            normalize='all',
            normalize_raw=True,
            complement='all'
    ):
        self.data_root = data_root
        self.context = context
        self.num_mel = num_mel
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.complement = complement

        self.data_sets = dict()
        for machine_type in range(6):
            self.data_sets[machine_type] = dict()
            for machine_id in TRAINING_ID_MAP[machine_type]:
                self.data_sets[machine_type][machine_id] = (
                    MachineDataSet(
                        machine_type,
                        machine_id,
                        data_root=self.data_root,
                        mode='training',
                        context=self.context,
                        num_mel=self.num_mel,
                        n_fft=self.n_fft,
                        hop_size=self.hop_size,
                        normalize=normalize_raw
                    ),
                    MachineDataSet(
                        machine_type,
                        machine_id,
                        data_root=self.data_root,
                        mode='validation',
                        context=self.context,
                        num_mel=self.num_mel,
                        n_fft=self.n_fft,
                        hop_size=self.hop_size,
                        normalize=normalize_raw
                    )
                )

        if normalize == 'all':
            data = []
            for machine_type in range(6):
                for machine_id in TRAINING_ID_MAP[machine_type]:
                    train, _ = self.data_sets[machine_type][machine_id]
                    data.append(train.data)
            data = np.concatenate(data, axis=1)
            mean = data.mean(axis=1, keepdims=True)
            std = data.std(axis=1, keepdims=True)
            for machine_type in range(6):
                for machine_id in TRAINING_ID_MAP[machine_type]:
                    train, val = self.data_sets[machine_type][machine_id]
                    train.data = (train.data - mean) / std
                    val.data = (val.data - mean) / std

        elif normalize == 'per_machine_type':
            for machine_type in range(6):
                data = []
                for machine_id in TRAINING_ID_MAP[machine_type]:
                    train, _ = self.data_sets[machine_type][machine_id]
                    data.append(train.data)
                data = np.concatenate(data, axis=1)
                mean = data.mean(axis=1, keepdims=True)
                std = data.std(axis=1, keepdims=True)
                for machine_id in TRAINING_ID_MAP[machine_type]:
                    train, val = self.data_sets[machine_type][machine_id]
                    train.data = (train.data - mean) / std
                    val.data = (val.data - mean) / std

        elif normalize == 'per_machine_id':
            for machine_type in range(6):
                for machine_id in TRAINING_ID_MAP[machine_type]:
                    train, val = self.data_sets[machine_type][machine_id]
                    data = train.data
                    mean = data.mean(axis=1, keepdims=True)
                    std = data.std(axis=1, keepdims=True)
                    train.data = (train.data - mean) / std
                    val.data = (val.data - mean) / std

        elif normalize == 'per_mic':
            data = []
            for machine_type in [0, 1, 2, 5]:
                for machine_id in TRAINING_ID_MAP[machine_type]:
                    train, _ = self.data_sets[machine_type][machine_id]
                    data.append(train.data)
            data = np.concatenate(data, axis=1)
            mean = data.mean(axis=1, keepdims=True)
            std = data.std(axis=1, keepdims=True)
            for machine_type in [0, 1, 2, 5]:
                for machine_id in TRAINING_ID_MAP[machine_type]:
                    train, val = self.data_sets[machine_type][machine_id]
                    train.data = (train.data - mean) / std
                    val.data = (val.data - mean) / std
            data = []
            for machine_type in [3, 4]:
                for machine_id in TRAINING_ID_MAP[machine_type]:
                    train, _ = self.data_sets[machine_type][machine_id]
                    data.append(train.data)
            data = np.concatenate(data, axis=1)
            mean = data.mean(axis=1, keepdims=True)
            std = data.std(axis=1, keepdims=True)
            for machine_type in [3, 4]:
                for machine_id in TRAINING_ID_MAP[machine_type]:
                    train, val = self.data_sets[machine_type][machine_id]
                    train.data = (train.data - mean) / std
                    val.data = (val.data - mean) / std
        elif normalize == 'none':
            pass
        else:
            raise AttributeError

    @property
    def observation_shape(self) -> tuple:
        return 1, self.num_mel, self.context

    def training_data_set(self, type, id):
        return self.data_sets[type][id][0]

    def validation_data_set(self, type, id):
        return self.data_sets[type][id][1]

    def complement_data_set(self, type, id):
        complement_sets = []
        if self.complement == 'all':
            for machine_type in range(6):
                for machine_id in TRAINING_ID_MAP[machine_type]:
                    if machine_type != type or machine_id != id:
                            complement_sets.append(self.data_sets[machine_type][machine_id][0])

        elif self.complement == 'same_mic_diff_type':
            if type in [3, 4]:
                types = [3, 4]
            else:
                types = [0, 1, 2, 5]
            for machine_type in types:
                for machine_id in TRAINING_ID_MAP[machine_type]:
                    if machine_type != type:
                        complement_sets.append(self.data_sets[machine_type][machine_id][0])

        elif self.complement == 'same_mic':
            if type in [3, 4]:
                types = [3, 4]
            else:
                types = [0, 1, 2, 5]
            for machine_type in types:
                for machine_id in TRAINING_ID_MAP[machine_type]:
                    if machine_type != type or machine_id != id:
                        complement_sets.append(self.data_sets[machine_type][machine_id][0])

        return torch.utils.data.ConcatDataset(complement_sets)


class MixUpDataSet(torch.utils.data.Dataset):

    def __init__(
            self,
            data_sets
    ):
        self.data_sets = data_sets

    def __getitem__(self, item):
        ds_1 = self.data_sets[np.random.random_integers(0, len(self.data_sets)-1)]
        sample_1 = ds_1[np.random.random_integers(0, len(ds_1)-1)]
        ds_2 = self.data_sets[np.random.random_integers(0, len(self.data_sets)-1)]
        sample_2 = ds_2[np.random.random_integers(0, len(ds_2)-1)]

        l = np.random.beta(1, 1, size=1).astype(np.float32)

        sample_1['observations'] = l * sample_1['observations'] + (1 - l) * sample_2['observations']

        return sample_1

    def __len__(self):
        return 20000


class MachineDataSet(torch.utils.data.Dataset):

    def __init__(
            self,
            machine_type,
            machine_id,
            data_root=os.path.join(os.path.expanduser('~'), 'shared', 'dcase2020_task2'),
            mode='training',
            context=5,
            num_mel=128,
            n_fft=1024,
            hop_size=512,
            normalize=True
    ):

        assert mode in ['training', 'validation', 'testing']

        self.num_mel = num_mel
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.normalize=normalize
        self.mode = mode
        self.data_root = data_root
        self.context = context
        self.machine_type = INVERSE_CLASS_MAP[machine_type]
        self.machine_id = machine_id

        if mode == 'training':
            files = glob.glob(os.path.join(data_root, 'dev_data', self.machine_type, 'train', '*_id_{:02d}_*.wav'.format(machine_id)))
        elif mode == 'validation':
            files = glob.glob(os.path.join(data_root, 'dev_data', self.machine_type, 'test', '*_id_{:02d}_*.wav'.format(machine_id)))
        elif mode == 'testing':
            raise NotImplementedError
        else:
            raise AttributeError
        assert len(files) > 0

        files = sorted(files)
        self.files = files
        self.file_length = self.__load_preprocess_file__(files[0]).shape[-1]
        self.num_samples_per_file = self.file_length - self.context + 1
        self.meta_data = self.__load_meta_data__(files)
        self.data = self.__load_data__(files)

    def __getitem__(self, item):
        # get offset in audio file
        offset = item % self.num_samples_per_file
        # get audio file index
        item = item // self.num_samples_per_file
        # load audio file and extract audio junk
        offset = item * self.file_length + offset
        observation = self.data[:, offset:offset + self.context]
        # create data object
        meta_data = self.meta_data[item].copy()
        meta_data['observations'] = observation[None]
        meta_data['file_ids'] = item

        return meta_data

    def __len__(self):
        return len(self.files) * self.num_samples_per_file

    def __load_meta_data__(self, files):
        data = []
        for f in files:
            md = self.__get_meta_data__(f)
            data.append(md)
        return data

    def __load_data__(self, files):
        file_name = "{}_{}_{}_{}_{}_{}_{}_{}.npy".format(
            self.num_mel,
            self.n_fft,
            self.hop_size,
            self.mode,
            self.context,
            self.machine_type,
            self.machine_id,
            self.normalize
        )
        file_path = os.path.join(self.data_root, file_name)

        if os.path.exists(file_path):
            print('Loading {} data set for machine type {} id {}...'.format(self.mode, self.machine_type, self.machine_id))
            data = np.load(file_path)
        else:
            print('Loading & saving {} data set for machine type {} id {}...'.format(self.mode, self.machine_type, self.machine_id))
            data = np.empty((self.num_mel, self.file_length * len(files)), dtype=np.float32)
            for i, f in enumerate(files):
                data[:, i * self.file_length:(i + 1) * self.file_length] = self.__load_preprocess_file__(f)
            np.save(file_path, data)

        return data

    def __load_preprocess_file__(self, file):
        x, sr = librosa.load(file, sr=None)
        if self.normalize:
            x = ((x - x.mean()) / x.std())

        x = librosa.feature.melspectrogram(
            y=x,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            n_mels=self.num_mel,
            power=1.0
        )

        x = 20.0 / 2.0 * np.log10(x + sys.float_info.epsilon)

        return x

    def __get_meta_data__(self, file_path):
        meta_data = os.path.split(file_path)[-1].split('_')
        machine_type = os.path.split(os.path.split(os.path.split(file_path)[0])[0])[1]
        machine_type = CLASS_MAP[machine_type]
        if len(meta_data) == 4:
            y = 0 if meta_data[0] == 'normal' else 1
            id = self.machine_id
            part = int(meta_data[3].split('.')[0])
        elif len(meta_data) == 3:
            y = -1
            id = self.machine_id
            part = int(meta_data[2].split('.')[0])
        else:
            raise AttributeError

        return {
            'targets': y,
            'machine_types': machine_type,
            'machine_ids': id,
            'part_numbers': part
        }
