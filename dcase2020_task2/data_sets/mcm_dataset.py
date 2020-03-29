import os
import torch.utils.data
import glob
import data_sets
import librosa
import sys
import numpy as np

CLASS_MAP = {
    'fan': 0,
    'pump': 1,
    'slider': 2,
    'ToyCar': 3,
    'ToyConveyor': 4,
    'valve': 5
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
    ):
        self.data_root = data_root
        self.context = context
        self.num_mel = num_mel
        self.n_fft = n_fft
        self.hop_size = hop_size

        self.training_data_sets = [
            MachineTypeDataSet(
                data_root=self.data_root,
                mode='training',
                context=self.context,
                machine_type=i,
                num_mel=self.num_mel,
                n_fft=self.n_fft,
                hop_size=self.hop_size
            ) for i in range(6)
        ]

        self.validation_data_sets = [
            MachineTypeDataSet(
                data_root=self.data_root,
                mode='validation',
                context=self.context,
                machine_type=i,
                num_mel=self.num_mel,
                n_fft=self.n_fft,
                hop_size=self.hop_size
            ) for i in range(6)
        ]

        self.testing_data_sets = [
            None  # TODO: add test set
            for i in range(6)
        ]

        for train, val, test in zip(self.training_data_sets, self.validation_data_sets, self.testing_data_sets):
            mean = train.data.mean(axis=1, keepdims=True)
            std = train.data.std(axis=1, keepdims=True)
            train.data = (train.data - mean) / std
            val.data = (val.data - mean) / std
            # TODO: add normalization
            # test.data = (test.data - mean) / std

    @property
    def observation_shape(self) -> tuple:
        return 1, self.num_mel, self.context

    def training_data_set(self, index):
        return self.training_data_sets[index]

    def validation_data_set(self, index):
        return self.validation_data_sets[index]

    def testing_data_set(self, index):
        return None

    def complement_data_set(self, index):
        complement_indices = list(range(6))
        complement_indices.pop(index)
        return torch.utils.data.ConcatDataset([self.training_data_sets[i] for i in complement_indices])


class MachineTypeDataSet(torch.utils.data.Dataset):

    def __init__(
            self,
            data_root=os.path.join(os.path.expanduser('~'), 'shared', 'dcase2020_task2'),
            mode='training',
            context=5,
            machine_type=0,
            num_mel=128,
            n_fft=1024,
            hop_size=512,
    ):

        assert mode in ['training', 'validation', 'testing']

        self.num_mel = num_mel
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.mode = mode
        self.data_root = data_root
        self.context = context
        self.machine_type = INVERSE_CLASS_MAP[machine_type]

        if mode == 'training':
            files = glob.glob(os.path.join(data_root, 'dev_data', self.machine_type, 'train', '*.wav'))
        elif mode == 'validation':
            files = glob.glob(os.path.join(data_root, 'dev_data', self.machine_type, 'test', '*.wav'))
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
        file_name = "{}_{}_{}_{}_{}_{}.npy".format(
            self.num_mel,
            self.n_fft,
            self.hop_size,
            self.mode,
            self.context,
            self.machine_type
        )
        file_path = os.path.join(self.data_root, file_name)

        if os.path.exists(file_path):
            print('Loading {} data set for machine type {}...'.format(self.mode, self.machine_type))
            data = np.load(file_path)
        else:
            print('Loading & saving {} data set for machine type {}...'.format(self.mode, self.machine_type))
            data = np.empty((self.num_mel, self.file_length * len(files)), dtype=np.float32)
            for i, f in enumerate(files):
                data[:, i * self.file_length:(i + 1) * self.file_length] = self.__load_preprocess_file__(f)
            np.save(file_path, data)

        return data

    def __load_preprocess_file__(self, file):
        x, sr = librosa.load(file, sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=x,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            n_mels=self.num_mel,
            power=2.0
        )
        log_mel_spectrogram = 20.0 / 2.0 * np.log10(mel_spectrogram + sys.float_info.epsilon)
        return log_mel_spectrogram

    @staticmethod
    def __get_meta_data__(file_path):
        meta_data = os.path.split(file_path)[-1].split('_')
        machine_type = os.path.split(os.path.split(os.path.split(file_path)[0])[0])[1]
        machine_type = CLASS_MAP[machine_type]
        if len(meta_data) == 4:
            y = 0 if meta_data[0] == 'normal' else 1
            id = int(meta_data[2])
            part = int(meta_data[3].split('.')[0])
        elif len(meta_data) == 3:
            y = -1
            id = int(meta_data[1])
            part = int(meta_data[2].split('.')[0])
        else:
            raise AttributeError

        return {
            'targets': y,
            'machine_types': machine_type,
            'machine_ids': id,
            'part_numbers': part
        }
