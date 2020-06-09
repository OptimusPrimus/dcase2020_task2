import torch.nn
import torch
from dcase2020_task2.models.custom import ACTIVATION_DICT, init_weights


class ResidualBlock(torch.nn.Module):

    def __init__(self, input_size, layer_size, kernel_sizes=[3, 1], batch_norm=False, activation='relu'):
        super().__init__()

        net = []
        for i, kernel_size in enumerate(kernel_sizes):
            net.append(
                torch.nn.Conv2d(input_size if i == 0 else layer_size, layer_size, kernel_size=kernel_size, padding=kernel_size // 2)
            )
            if batch_norm:
                net.append(torch.nn.BatchNorm2d(layer_size))
            net.append(ACTIVATION_DICT[activation]())

        if input_size != layer_size:
            self.linear_transform = torch.nn.Conv2d(input_size, layer_size, kernel_size=(1, 1), padding=(0, 0))
        else:
            self.linear_transform = None

        self.last_activation = net.pop()
        self.net = torch.nn.Sequential(*net)

    def forward(self, x):
        if self.linear_transform:
            x_ = self.linear_transform(x)
        else:
            x_ = x
        return self.last_activation(self.net(x) + x_)


class ResNet(torch.nn.Module):

    def __init__(
            self,
            input_shape,
            num_outputs=1,
            activation='relu',
            batch_norm=False,
            **kwargs

    ):
        super().__init__()

        self.input_shape = input_shape

        net = [torch.nn.Conv2d(self.input_shape[0], 64, kernel_size=(5, 5), padding=(2, 2))]
        if batch_norm:
            net.append(torch.nn.BatchNorm2d(64))
        net.append(ACTIVATION_DICT[activation]())

        net.append(ResidualBlock(64, 128, kernel_sizes=[3, 3], batch_norm=batch_norm))
        net.append(torch.nn.MaxPool2d(2))

        net.append(ResidualBlock(128, 256, kernel_sizes=[3, 3], batch_norm=batch_norm))
        net.append(torch.nn.MaxPool2d(2))

        net.append(ResidualBlock(256, 512, kernel_sizes=[3, 3], batch_norm=batch_norm))
        net.append(torch.nn.MaxPool2d(2))

        net.append(torch.nn.Conv2d(512, num_outputs, kernel_size=(3, 3)))
        if batch_norm:
            net.append(torch.nn.BatchNorm2d(num_outputs))

        net.append(torch.nn.AdaptiveMaxPool2d(1))

        self.net = torch.nn.Sequential(
            *net
        )

    def forward(self, batch):
        x = batch['observations']
        batch['scores'] = self.net(x).view(-1, 1)
        return batch