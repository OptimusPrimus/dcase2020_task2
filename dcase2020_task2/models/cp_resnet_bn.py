# coding: utf-8
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

from librosa.filters import mel as librosa_mel_fn


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity="relu")
        # nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


layer_index_total = 0


def initialize_weights_fixup(module):
    if isinstance(module, AttentionAvg):
        print("AttentionAvg init..")
        module.forw_conv[0].weight.data.zero_()
        module.atten[0].bias.data.zero_()
        nn.init.kaiming_normal_(module.atten[0].weight.data, mode='fan_in', nonlinearity="sigmoid")
    if isinstance(module, BasicBlock):
        # He init, rescaled by Fixup multiplier
        b = module
        n = b.conv1.kernel_size[0] * b.conv1.kernel_size[1] * b.conv1.out_channels
        print(b.layer_index, math.sqrt(2. / n), layer_index_total ** (-0.5))
        b.conv1.weight.data.normal_(0, (layer_index_total ** (-0.5)) * math.sqrt(2. / n))
        b.conv2.weight.data.zero_()
        if b.shortcut._modules.get('conv') is not None:
            convShortcut = b.shortcut._modules.get('conv')
            n = convShortcut.kernel_size[0] * convShortcut.kernel_size[1] * convShortcut.out_channels
            convShortcut.weight.data.normal_(0, math.sqrt(2. / n))
    if isinstance(module, nn.Conv2d):
        pass
        # nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity="relu")
        # nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


first_RUN = True


def calc_padding(kernal):
    try:
        return kernal // 3
    except TypeError:
        return [k // 3 for k in kernal]


class AttentionAvg(nn.Module):

    def __init__(self, in_channels, out_channels, sum_all=True):
        super(AttentionAvg, self).__init__()
        self.sum_dims = [2, 3]
        if sum_all:
            self.sum_dims = [1, 2, 3]
        self.forw_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.atten = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        a1 = self.forw_conv(x)
        atten = self.atten(x)
        num = atten.size(2) * atten.size(3)
        asum = atten.sum(dim=self.sum_dims, keepdim=True) + 1e-8
        return a1 * atten * num / asum


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, k1=3, k2=3):
        super(BasicBlock, self).__init__()
        global layer_index_total
        self.layer_index = layer_index_total
        layer_index_total = layer_index_total + 1
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=k1,
            stride=stride,  # downsample with first conv
            padding=calc_padding(k1),
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=k2,
            stride=1,
            padding=calc_padding(k2),
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckBlock, self).__init__()

        bottleneck_channels = out_channels // self.expansion

        self.conv1 = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,  # downsample with 3x3 conv
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()  # identity
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))  # not apply ReLU
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()

        input_shape = config['input_shape']
        n_classes = config['n_classes']

        base_channels = config['base_channels']
        block_type = config['block_type']
        depth = config['depth']
        self.pooling_padding = config.get("pooling_padding", 0) or 0
        self.use_raw_spectograms = config.get("use_raw_spectograms") or False
        self.apply_softmax = config.get("apply_softmax") or False

        assert block_type in ['basic', 'bottleneck']
        if self.use_raw_spectograms:
            mel_basis = librosa_mel_fn(
                22050, 2048, 256)
            mel_basis = torch.from_numpy(mel_basis).float()
            self.register_buffer('mel_basis', mel_basis)
        if block_type == 'basic':
            block = BasicBlock
            n_blocks_per_stage = (depth - 2) // 6
            assert n_blocks_per_stage * 6 + 2 == depth
        else:
            block = BottleneckBlock
            n_blocks_per_stage = (depth - 2) // 9
            assert n_blocks_per_stage * 9 + 2 == depth
        n_blocks_per_stage = [n_blocks_per_stage, n_blocks_per_stage, n_blocks_per_stage]

        if config.get("n_blocks_per_stage") is not None:
            print("n_blocks_per_stage is specified ignoring the depth param, nc=" + str(config.get("n_channels")))
            n_blocks_per_stage = config.get("n_blocks_per_stage")

        n_channels = config.get("n_channels")
        if n_channels is None:
            n_channels = [
                base_channels,
                base_channels * 2 * block.expansion,
                base_channels * 4 * block.expansion
            ]
        if config.get("grow_a_lot"):
            n_channels[2] = base_channels * 8 * block.expansion

        self.in_c = nn.Sequential(nn.Conv2d(
            input_shape[1],
            n_channels[0],
            kernel_size=5,
            stride=2,
            padding=1,
            bias=False),
            nn.BatchNorm2d(n_channels[0]),
            nn.ReLU(True)
        )
        self.stage1 = self._make_stage(
            n_channels[0], n_channels[0], n_blocks_per_stage[0], block, stride=1, maxpool=config['stage1']['maxpool'],
            k1s=config['stage1']['k1s'], k2s=config['stage1']['k2s'])
        self.stage2 = self._make_stage(
            n_channels[0], n_channels[1], n_blocks_per_stage[1], block, stride=1, maxpool=config['stage2']['maxpool'],
            k1s=config['stage2']['k1s'], k2s=config['stage2']['k2s'])
        self.stage3 = self._make_stage(
            n_channels[1], n_channels[2], n_blocks_per_stage[2], block, stride=1, maxpool=config['stage3']['maxpool'],
            k1s=config['stage3']['k1s'], k2s=config['stage3']['k2s'])

        ff_list = []
        if config.get("attention_avg"):
            if config.get("attention_avg") == "sum_all":
                ff_list.append(AttentionAvg(n_channels[2], n_classes, sum_all=True))
            else:
                ff_list.append(AttentionAvg(n_channels[2], n_classes, sum_all=False))
        else:
            ff_list += [nn.Conv2d(
                n_channels[2],
                n_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
                nn.BatchNorm2d(n_classes),
            ]

        self.stop_before_global_avg_pooling = False
        if config.get("stop_before_global_avg_pooling"):
            self.stop_before_global_avg_pooling = True
        else:
            ff_list.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.feed_forward = nn.Sequential(
            *ff_list
        )
        # # compute conv feature size
        # with torch.no_grad():
        #     self.feature_size = self._forward_conv(
        #         torch.zeros(*input_shape)).view(-1).shape[0]
        #
        # self.fc = nn.Linear(self.feature_size, n_classes)

        # initialize weights
        if config.get("weight_init") == "fixup":
            self.apply(initialize_weights)
            if isinstance(self.feed_forward[0], nn.Conv2d):
                self.feed_forward[0].weight.data.zero_()
            self.apply(initialize_weights_fixup)
        else:
            self.apply(initialize_weights)
        self.use_check_point = config.get("use_check_point") or False

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride, maxpool=set(), k1s=[3, 3, 3, 3, 3, 3],
                    k2s=[3, 3, 3, 3, 3, 3]):
        stage = nn.Sequential()
        if 0 in maxpool:
            stage.add_module("maxpool{}_{}".format(0, 0)
                             , nn.MaxPool2d(2, 2, padding=self.pooling_padding))
        for index in range(n_blocks):
            stage.add_module('block{}'.format(index + 1),
                             block(in_channels,
                                   out_channels,
                                   stride=stride, k1=k1s[index], k2=k2s[index]))

            in_channels = out_channels
            stride = 1
            # if index + 1 in maxpool:
            for m_i, mp_pos in enumerate(maxpool):
                if index + 1 == mp_pos:
                    stage.add_module("maxpool{}_{}".format(index + 1, m_i)
                                     , nn.MaxPool2d(2, 2, padding=self.pooling_padding))
        return stage

    def _forward_conv(self, x):
        global first_RUN

        if first_RUN: print("x:", x.size())
        x = self.in_c(x)
        if first_RUN: print("in_c:", x.size())

        if self.use_check_point:
            if first_RUN: print("use_check_point:", x.size())
            return checkpoint_sequential([self.stage1, self.stage2, self.stage3], 3,
                                         (x))
        x = self.stage1(x)

        if first_RUN: print("stage1:", x.size())
        x = self.stage2(x)
        if first_RUN: print("stage2:", x.size())
        x = self.stage3(x)
        if first_RUN: print("stage3:", x.size())
        return x

    def forward(self, x):
        global first_RUN
        if self.use_raw_spectograms:
            if first_RUN: print("raw_x:", x.size())
            x = torch.log10(torch.sqrt((x * x).sum(dim=3)))
            if first_RUN: print("log10_x:", x.size())
            x = torch.matmul(self.mel_basis, x)
            if first_RUN: print("mel_basis_x:", x.size())
            x = x.unsqueeze(1)
        x = self._forward_conv(x)
        x = self.feed_forward(x)
        if first_RUN: print("feed_forward:", x.size())
        if self.stop_before_global_avg_pooling:
            first_RUN = False
            return x
        logit = x.squeeze(2).squeeze(2)

        if first_RUN: print("logit:", logit.size())
        if self.apply_softmax:
            logit = torch.softmax(logit, 1)
        first_RUN = False
        return logit