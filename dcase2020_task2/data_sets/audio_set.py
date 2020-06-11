import os
import torch.utils.data
import glob
from dcase2020_task2.data_sets import BaseDataSet, CLASS_MAP, INVERSE_CLASS_MAP, TRAINING_ID_MAP, EVALUATION_ID_MAP, ALL_ID_MAP,\
    enumerate_development_datasets, enumerate_evaluation_datasets
import librosa
import numpy as np
from dcase2020_task2.data_sets import MCMDataSet
import pickle


class AudioSet(BaseDataSet):

    def __init__(
            self,
            data_root=os.path.join(os.path.expanduser('~'), 'shared', 'audioset', 'audiosetdata'),
            context=5,
            num_mel=128,
            n_fft=1024,
            hop_size=512,
            power=2.0,
            fmin=0,
            normalize_raw=True,
            hop_all=False
    ):
        self.data_root = data_root
        self.context = context
        self.num_mel = num_mel
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.power = power
        self.fmin = fmin
        self.hop_all = hop_all

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

        class_names = sorted([class_name for class_name in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, class_name))])[:30]

        training_sets = []
        data_arrays = []
        for class_name in class_names:
            training_sets.append(AudioSetClassSubset(class_name, **kwargs))
            data = training_sets[-1].data
            for i, file in enumerate(data):
                data_arrays.append(file)
        data_arrays = np.concatenate(data_arrays, axis=-1)

        self.training_set = torch.utils.data.ConcatDataset(training_sets)
        self.validation_set = None
        self.mean = data_arrays.mean(axis=1, keepdims=True)
        self.std = data_arrays.std(axis=1, keepdims=True)

    @property
    def observation_shape(self) -> tuple:
        return 1, self.num_mel, self.context

    def training_data_set(self):
        return self.training_set

    def validation_data_set(self):
        return self.validation_set

    def mean_std(self):
        return self.mean, self.std


class AudioSetClassSubset(torch.utils.data.Dataset):

    def __init__(
            self,
            class_name,
            data_root=os.path.join(os.path.expanduser('~'), 'shared', 'audioset', 'audiosetdata'),
            context=5,
            num_mel=128,
            n_fft=1024,
            hop_size=512,
            power=2.0,
            normalize=True,
            fmin=0,
            hop_all=False,
            max_file_length=350
    ):

        self.num_mel = num_mel
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.power = power
        self.normalize = normalize
        self.data_root = data_root
        self.context = context
        self.fmin = fmin
        self.hop_all = hop_all
        self.class_name = class_name
        self.max_file_length = max_file_length

        files = glob.glob(os.path.join(data_root, class_name, '*.wav'))

        assert len(files) > 0

        files = sorted(files)
        self.files = files

        self.meta_data = self.__load_meta_data__(files)
        self.data = self.__load_data__(files)

        self.index_map = {}
        ctr = 0
        for i, file in enumerate(self.data):
            for j in range(file.shape[-1] + 1 - context):
                self.index_map[ctr] = (i, j)
                ctr += 1
        self.length = ctr

    def __getitem__(self, item):
        file_idx, offset = self.index_map[item]
        observation = self.data[file_idx][:, offset:offset + self.context]
        meta_data = self.meta_data[file_idx].copy()
        meta_data['observations'] = observation[None]

        return meta_data

    def __len__(self):
        return self.length

    def __load_meta_data__(self, files):
        data = []
        for f in files:
            md = self.__get_meta_data__(f)
            data.append(md)
        return data

    def __load_data__(self, files):
        file_name = "{}_{}_{}_{}_{}_{}_{}.npz".format(
            self.num_mel,
            self.n_fft,
            self.hop_size,
            self.power,
            self.normalize,
            self.fmin,
            self.class_name
        )
        file_path = os.path.join(self.data_root, file_name)

        data = []
        if os.path.exists(file_path):
            print('Loading audio set class {} '.format(self.class_name))
            container = np.load(file_path)
            data = [container[key] for key in container]
        else:
            print('Loading & saving audio set class {} '.format(self.class_name))
            for i, f in enumerate(files):
                file = self.__load_preprocess_file__(f)
                if file.shape[1] > self.max_file_length:
                    print(f'File too long: {f} - {file.shape[1]}')
                    file = file[:, :self.max_file_length]
                data.append(file)
            np.savez(file_path, *data)
        return data

    def __load_preprocess_file__(self, file):
        x, sr = librosa.load(file, sr=16000, mono=True)
        if len(x) > (self.max_file_length + 1 * self.hop_size) + self.n_fft:
            x = x[:(self.max_file_length + 1) * self.hop_size + self.n_fft]
        if self.normalize:
            x = (x - x.mean()) / x.std()

        x = librosa.feature.melspectrogram(
            y=x,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            n_mels=self.num_mel,
            power=self.power,
            fmin=self.fmin
        )

        if self.power == 1:
            x = librosa.core.amplitude_to_db(x)
        elif self.power == 2:
            x = librosa.core.power_to_db(x)
        else:
            raise AttributeError

        return x

    def __get_meta_data__(self, file_path):
        return {
            'targets': 1,
            'machine_types': -1,
            'machine_ids': -1,
            'file_ids': os.sep.join(os.path.normpath(file_path).split(os.sep)[-4:])
        }

if __name__ == '__main__':
    mcmc = MCMDataSet(0, 0)
    a = audio_set = AudioSet(
        normalize=(mcmc.mean, mcmc.std)
    ).training_data_set()[0]

    print(a)



