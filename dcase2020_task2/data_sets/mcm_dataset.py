import os
import torch.utils.data
import glob
import tqdm
import data_sets
import librosa
import sys
import numpy as np


class MCMDataset(torch.utils.data.Dataset, data_sets.BaseDataSet):

    def __init__(
            self,
            data_root=os.path.join(os.path.expanduser('~'), 'shared', 'dcase2020_task2'),
            mode='training',
            context=5,
            machine_type=0,
            num_mel=128,
            n_fft=1024,
            hop_size=512
    ):

        assert mode in ['training', 'validation', 'testing']

        self.num_mel = num_mel
        self.n_fft = n_fft
        self.hop_size = hop_size

        self.mode = mode
        self.data_root = data_root
        self.context = context
        self.machine_type = self.inverse_class_map[machine_type]

        if mode == 'training':
            files = glob.glob(os.path.join(data_root, 'dev_data', self.machine_type, 'train', '*.wav'))
        elif mode == 'validation':
            files = glob.glob(os.path.join(data_root, 'dev_data', self.machine_type, 'test', '*.wav'))
        elif mode == 'testing':
            raise NotImplementedError
        else:
            raise AttributeError

        assert len(files) > 0

        self.files = files
        self.file_length = None
        self.num_samples_per_file = None
        self.mean = None
        self.std = None
        self.meta_data = self.__load_meta_data__(sorted(files))
        self.data = self.__load_data__(sorted(files))

        self.complement_data_set = None


    @property
    def class_map(self):
        return {
            'fan': 0,
            'pump': 1,
            'slider': 2,
            'ToyCar': 3,
            'ToyConveyor': 4,
            'valve': 5
        }

    @property
    def inverse_class_map(self):
        return {
            0: 'fan',
            1: 'pump',
            2: 'slider',
            3: 'ToyCar',
            4: 'ToyConveyor',
            5: 'valve'
        }

    @property
    def observation_shape(self) -> tuple:
        return 1, self.num_mel, self.context

    def get_complement_data_set(self):

        if self.complement_data_set:
            return self.complement_data_set

        machine_types = list(range(6))
        machine_types.remove(self.class_map[self.machine_type])
        data_sets = [
            MCMDataset(
                data_root=self.data_root,
                mode=self.mode,
                context=self.context,
                machine_type=i,
                num_mel=self.num_mel,
                n_fft=self.n_fft,
                hop_size=self.hop_size
            ) for i in machine_types
        ]

        for data_set in data_sets:
            data_set.mean = self.mean
            data_set.std = self.std

        data_set = torch.utils.data.ConcatDataset(data_sets)
        self.complement_data_set = data_set
        return data_set

    def get_various_data_set(self):

        data_sets = [
            self.get_complement_data_set(),
            self
        ]

        for data_set in data_sets:
            data_set.mean = self.mean
            data_set.std = self.std

        data_set = torch.utils.data.ConcatDataset(data_sets)
        return data_set

    def __getitem__(self, item):
        # get offset in audio file
        offset = item % self.num_samples_per_file
        # get audio file index
        item = item // self.num_samples_per_file
        # load audio file and extract audio junk
        offset = item*self.file_length+offset
        observation = self.data[:, offset:offset+self.context]
        # create data object
        meta_data = self.meta_data[item].copy()
        meta_data['observations'] = self.__normalize_observation__(observation)[None]
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
        self.file_length = self.__load_preprocess_file__(files[0]).shape[-1]
        self.num_samples_per_file = self.file_length - self.context + 1
        data = np.empty((self.num_mel, self.file_length*len(files)), dtype=np.float32)
        for i, f in tqdm.tqdm(enumerate(files), total=len(files)):
            data[:, i * self.file_length:(i+1) * self.file_length] = self.__load_preprocess_file__(f)
        self.mean = data.mean(axis=1)
        self.std = data.std(axis=1)
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

    def __get_meta_data__(self, file_path):
        meta_data = os.path.split(file_path)[-1].split('_')
        machine_type = os.path.split(os.path.split(os.path.split(file_path)[0])[0])[1]
        machine_type = self.class_map[machine_type]
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

    def __normalize_observation__(self, x):
        return (x - self.mean[:, None]) / self.std[:, None]
