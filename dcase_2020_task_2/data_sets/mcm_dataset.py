import os
import torch.utils.data
import torchaudio
import glob
import tqdm
import data_sets


class MCMDataset(torch.utils.data.Dataset, data_sets.BaseDataSet):

    def __init__(
            self,
            data_root=os.path.join(os.path.expanduser('~'), 'shared', 'DCASE_Task_2'),
            mode='training',
            context=5,
            transformations=[
                torchaudio.transforms.MelScale(n_mels=128, sample_rate=16000, f_min=0, f_max=None)
            ]
    ):

        assert mode in ['training', 'validation', 'testing']

        self.mode = mode
        self.data_root = data_root
        self.context = context
        self.transformations = transformations

        if mode == 'training':
            files = glob.glob(os.path.join(data_root, 'dev_data', '*', 'train', '*.wav'))
        elif mode == 'validation':
            files = glob.glob(os.path.join(data_root, 'dev_data', '*', 'test', '*.wav'))
        elif mode == 'testing':
            raise NotImplementedError
        else:
            raise AttributeError

        self.data = self.__load_data__(sorted(files))
        self.num_sampels_per_file = self.data[-1]['observation'].shape[2] // self.context

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

    def get_subset(self, machine_types):
        if type(machine_types) is int:
            machine_types = [machine_types]
        indices = []
        for i, sample in enumerate(self.data):
            if sample['machine_type'] in machine_types:
                indices += list(range(i*self.num_sampels_per_file, (i+1)*self.num_sampels_per_file))
        return torch.utils.data.Subset(self, indices)

    def __getitem__(self, item):
        # get offset in audio file
        offset = item % self.num_sampels_per_file
        # get audio file index
        item = item // self.num_sampels_per_file
        # load audio file and extract audio junk
        sample = self.data[item]
        # create data object
        return {
            'observations': sample['observation'][:, :, offset * self.context: (offset + 1) * self.context],
            'targets': sample['target'],
            'machine_types': sample['machine_type'],
            'machine_ids': sample['machine_id'],
            'part_numbers': sample['part_number']
        }

    def __len__(self):
        return len(self.data) * self.num_sampels_per_file

    def __load_data__(self, files):
        data = []
        for f, md in tqdm.tqdm(zip(files, [self.__get_meta_data__(f) for f in files]), total=len(files)):
            md['observation'] = self.__load_preprocess_file__(f)
            data.append(md)
        return data

    def __load_preprocess_file__(self, file):
        x = torchaudio.load(file)[0]
        # stft
        x = torch.stft(x, 1024, hop_length=512, window=torch.hann_window(1024), center=False)
        # power spectrogram
        x = x.pow(2).sum(-1)
        # to mel scale
        for t in self.transformations:
            x = t(x)
        return x

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

