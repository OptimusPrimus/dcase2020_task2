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
            machine_type=0
    ):

        assert mode in ['training', 'validation', 'testing']

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

        self.data = self.__load_data__(sorted(files))
        self.num_sampels_per_file = self.data[0]['observation'].shape[2] - self.context + 1

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
        return (1, 128, self.context)

    def normalize_observations(self, x):
        x['normalized_observations'] = x['observations']
        return x

    def __getitem__(self, item):
        # get offset in audio file
        offset = item % self.num_sampels_per_file
        # get audio file index
        item = item // self.num_sampels_per_file
        # load audio file and extract audio junk
        sample = self.data[item]
        # create data object
        return {
            'observations': sample['observation'][:, :, offset: offset + self.context],
            'targets': sample['target'],
            'machine_types': sample['machine_type'],
            'machine_ids': sample['machine_id'],
            'part_numbers': sample['part_number'],
            'file_ids': item
        }

    def __len__(self):
        return len(self.data) * self.num_sampels_per_file

    def __load_data__(self, files):
        data = []
        for f in tqdm.tqdm(files):
            md = self.__get_meta_data__(f)
            md['observation'] = self.__load_preprocess_file__(f)
            data.append(md)
        return data

    @staticmethod
    def __load_preprocess_file__(file):

        x, sr = librosa.load(file, sr=None)

        mel_spectrogram = librosa.feature.melspectrogram(
            y=x,
            sr=sr,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            power=2.0
        )

        log_mel_spectrogram = 20.0 / 2.0 * np.log10(mel_spectrogram + sys.float_info.epsilon)

        return log_mel_spectrogram[None]

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
            'target': y,
            'machine_type': machine_type,
            'machine_id': id,
            'part_number': part
        }
