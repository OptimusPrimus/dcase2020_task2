import torch.nn
import torch
from dcase2020_task2.models.custom import ACTIVATION_DICT, init_weights
from dcase2020_task2.models.cp_resnet_bn import Network


class ResNet(torch.nn.Module):

    def __init__(
            self,
            input_shape,
            num_outputs=1,
            base_channels=128,
            rf='normal',
            **kwargs

    ):
        super().__init__()

        self.input_shape = input_shape

        configs = {
            'normal': {
                'arch': 'cp_resnet',
                'base_channels': base_channels,
                'block_type': 'basic',
                'depth': 26,
                'input_shape': (1, *input_shape),
                'multi_label': False,
                'n_blocks_per_stage': [4, 1, 2],
                'n_classes': num_outputs,
                'prediction_threshold': 0.4,
                'stage1': {'k1s': [3, 3, 3, 3], 'k2s': [1, 3, 3, 1], 'maxpool': [1, 2, 4]},
                'stage2': {'k1s': [1, 1, 1, 1], 'k2s': [1, 1, 1, 1], 'maxpool': []},
                'stage3': {'k1s': [1, 1, 1, 1], 'k2s': [1, 1, 1, 1], 'maxpool': []},
                'use_bn': True,
                'weight_init': 'fixup',
                'pooling_padding': None,
                'use_raw_spectograms': None,
                'apply_softmax': None,
                'n_channels': None,
                'grow_a_lot': None,
                'attention_avg': None,
                'stop_before_global_avg_pooling': None,
                'use_check_point': None
            },
            'very_small': {
                'arch': 'cp_resnet',
                'base_channels': base_channels,
                'block_type': 'basic',
                'depth': 26,
                'input_shape': (1, *input_shape),
                'multi_label': False,
                'n_blocks_per_stage': [4, 1, 2],
                'n_classes': num_outputs,
                'prediction_threshold': 0.4,
                'stage1': {'k1s': [3, 3, 3, 1], 'k2s': [1, 1, 1, 1], 'maxpool': [1, 2, 4]},
                'stage2': {'k1s': [1, 1, 1, 1], 'k2s': [1, 1, 1, 1], 'maxpool': []},
                'stage3': {'k1s': [1, 1, 1, 1], 'k2s': [1, 1, 1, 1], 'maxpool': []},
                'use_bn': True,
                'weight_init': 'fixup',
                'pooling_padding': None,
                'use_raw_spectograms': None,
                'apply_softmax': None,
                'n_channels': None,
                'grow_a_lot': None,
                'attention_avg': None,
                'stop_before_global_avg_pooling': None,
                'use_check_point': None
            },
            'a_bit_larger': {
                'arch': 'cp_resnet',
                'base_channels': base_channels,
                'block_type': 'basic',
                'depth': 26,
                'input_shape': (1, *input_shape),
                'multi_label': False,
                'n_blocks_per_stage': [4, 1, 2],
                'n_classes': num_outputs,
                'prediction_threshold': 0.4,
                'stage1': {'k1s': [3, 3, 3, 3], 'k2s': [1, 3, 3, 3], 'maxpool': [1, 2, 4]},
                'stage2': {'k1s': [1, 1, 1, 1], 'k2s': [1, 1, 1, 1], 'maxpool': []},
                'stage3': {'k1s': [1, 1, 1, 1], 'k2s': [1, 1, 1, 1], 'maxpool': []},
                'use_bn': True,
                'weight_init': 'fixup',
                'pooling_padding': None,
                'use_raw_spectograms': None,
                'apply_softmax': None,
                'n_channels': None,
                'grow_a_lot': None,
                'attention_avg': None,
                'stop_before_global_avg_pooling': None,
                'use_check_point': None
            },
        'a_bit_smaller': {
            'arch': 'cp_resnet',
            'base_channels': base_channels,
            'block_type': 'basic',
            'depth': 26,
            'input_shape': (1, *input_shape),
            'multi_label': False,
            'n_blocks_per_stage': [4, 1, 2],
            'n_classes': num_outputs,
            'prediction_threshold': 0.4,
            'stage1': {'k1s': [3, 3, 3, 3], 'k2s': [1, 3, 1, 1], 'maxpool': [1, 2, 4]},
            'stage2': {'k1s': [1, 1, 1, 1], 'k2s': [1, 1, 1, 1], 'maxpool': []},
            'stage3': {'k1s': [1, 1, 1, 1], 'k2s': [1, 1, 1, 1], 'maxpool': []},
            'use_bn': True,
            'weight_init': 'fixup',
            'pooling_padding': None,
            'use_raw_spectograms': None,
            'apply_softmax': None,
            'n_channels': None,
            'grow_a_lot': None,
            'attention_avg': None,
            'stop_before_global_avg_pooling': None,
            'use_check_point': None
        }
        }

        self.net = Network(configs[rf])

    def forward(self, batch):
        x = batch['observations']
        batch['scores'] = self.net(x).view(-1, 1)
        return batch
