"""
Created by Hamid Eghbal-zadeh at 1/2/20
Johannes Kepler University of Linz
"""
from torch import nn


class Convolution2dLayer(nn.Module):
    def __init__(self, *, in_channels, out_channels,
            stride=2, kernel_size=3, padding=1,
            activation=nn.ReLU(),
            preactivate=False,
            with_batchnorm=True):
        super().__init__()

        def make_tuple(i):
            if isinstance(i, int):
                return (i, i)
            else:
                return i

        self.stride = make_tuple(stride)
        self.kernel_size = make_tuple(kernel_size)
        self.padding = make_tuple(padding)
        self.out_channels = out_channels
        self.preactivate = preactivate

        if activation is None:
            activation = lambda x: x

        self.activation = activation

        self.with_batchnorm = with_batchnorm
        if with_batchnorm:
            self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels,
                out_channels=out_channels,
                stride=self.stride,
                kernel_size=self.kernel_size,
                padding=self.padding)

    def out_height(self, in_height):
        numerator = (in_height + 2 * self.padding[0] -
                (self.kernel_size[0] - 1) - 1)
        return (numerator // self.stride[0]) + 1

    def out_width(self, in_width):
        numerator = (in_width + 2 * self.padding[1] -
                (self.kernel_size[1] - 1) - 1)
        return (numerator // self.stride[1]) + 1

    def out_size(self, in_size):
        return (self.out_height(in_size[0]), self.out_width(in_size[1]))

    def forward(self, x):
        if self.with_batchnorm:
            x = self.bn(x)

        if self.preactivate:
            x = self.activation(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = self.activation(x)

        return x
# ===============================================================================================================================================



class FullyConnectedLayer(nn.Module):
    def __init__(self, in_size, out_size,
            activation=nn.ReLU(),
            with_batchnorm=True,
            dropout=0.0):
        super().__init__()

        self.with_batchnorm = with_batchnorm
        if with_batchnorm:
            self.bn = nn.BatchNorm1d(in_size)

        self.fc = nn.Linear(in_size, out_size)

        self._dropout_value = dropout
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

        self.activation = activation

    def forward(self, x_in):
        if self.with_batchnorm:
            x = self.bn(x_in)
        else:
            x = x_in

        x = self.fc(x)

        if self._dropout_value > 0.0:
            x = self.dropout(x)

        if self.activation is not None:
            x = self.activation(x)
        return x
# ===============================================================================================================================================



def make_tuple(i):
    if isinstance(i, int):
        return (i, i)
    else:
        return i


class Deconvolution2dLayer(nn.Module):
    def __init__(self, *, in_channels, out_channels,
            stride=2, kernel_size=3, padding=1,
            output_padding=1,
            activation=nn.ReLU(),
            preactivate=False,
            with_batchnorm=True):
        super().__init__()

        self.stride = make_tuple(stride)
        self.kernel_size = make_tuple(kernel_size)
        self.padding = make_tuple(padding)
        self.output_padding = make_tuple(output_padding)
        self.out_channels = out_channels
        self.preactivate = preactivate

        if activation is None:
            activation = lambda x: x

        self.activation = activation

        self.with_batchnorm = with_batchnorm
        if with_batchnorm:
            self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels,
                out_channels=out_channels,
                stride=self.stride,
                kernel_size=self.kernel_size,
                padding=self.padding,
                output_padding=output_padding)

    @classmethod
    def matching_input_and_output_size(cls, input_size, output_size, **kwargs):
        candidate = cls(**{**kwargs, 'output_padding': 0})
        h, w = candidate.out_size(input_size)
        output_padding = [output_size[0] - h, output_size[1] - w]
        padding = list(make_tuple(kwargs.get('padding', 0)))

        if output_padding[0] < 0:
            padding[0] += 1
            output_padding[0] += 2

        if output_padding[1] < 0:
            padding[1] += 1
            output_padding[1] += 2

        kwargs['padding'] = padding
        kwargs['output_padding'] = output_padding
        return cls(**kwargs)

    def out_height(self, in_height):
        return (in_height - 1) * self.stride[0] - 2 * self.padding[0] +\
               self.kernel_size[0] + self.output_padding[0]

    def out_width(self, in_width):
        return (in_width - 1) * self.stride[1] - 2 * self.padding[1] +\
               self.kernel_size[1] + self.output_padding[1]

    def out_size(self, in_size):
        return (self.out_height(in_size[0]), self.out_width(in_size[1]))

    def forward(self, x):
        if self.with_batchnorm:
            x = self.bn(x)

        if self.preactivate:
            x = self.activation(x)
            x = self.deconv(x)
        else:
            x = self.deconv(x)
            x = self.activation(x)

        return x
