import torch.nn
from dcase2020_task2.models import VAEBase  #
from dcase2020_task2.priors import NoPrior
import numpy as np
import torch

from dcase2020_task2.models.custom import ACTIVATION_DICT, init_weights


class AE(torch.nn.Module, VAEBase):

    def __init__(
            self,
            input_shape,
            reconstruction_loss,
            prior=None,
            hidden_size=128,
            num_hidden=3,
            activation='relu',
            batch_norm=False
    ):
        super().__init__()

        activation_fn = ACTIVATION_DICT[activation]
        if prior is None:
            prior = NoPrior(latent_size=hidden_size)
        self.input_shape = input_shape
        self.prior = prior
        self.reconstruction = reconstruction_loss

        # encoder sizes/ layers
        sizes = [np.prod(input_shape)] + [hidden_size] * num_hidden + [prior.input_size]
        encoder_layers = []
        for i, o in zip(sizes[:-1], sizes[1:]):
            encoder_layers.append(torch.nn.Linear(i, o))
            if batch_norm:
                encoder_layers.append(torch.nn.BatchNorm1d(o))
            encoder_layers.append(activation_fn())

        # symetric decoder sizes/ layers
        sizes = sizes[::-1]
        decoder_layers = []
        for i, o in zip(sizes[:-1], sizes[1:]):
            decoder_layers.append(torch.nn.Linear(i, o))
            if batch_norm:
                decoder_layers.append(torch.nn.BatchNorm1d(o))
            decoder_layers.append(activation_fn())
        # remove last relu
        _ = decoder_layers.pop()
        if batch_norm:
            _ = decoder_layers.pop()

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.decoder = torch.nn.Sequential(*decoder_layers)
        self.apply(init_weights)

    def forward(self, batch):
        batch = self.encode(batch)
        batch = self.prior(batch)
        batch = self.decode(batch)
        return batch

    def encode(self, batch):
        x = batch['observations']
        x = x.view(x.shape[0], -1)
        batch['pre_codes'] = self.encoder(x)
        return batch

    def decode(self, batch):
        batch['pre_reconstructions'] = self.decoder(batch['codes']).view(-1, *self.input_shape)
        batch = self.reconstruction(batch)
        return batch


class ConvlBlock(torch.nn.Module):

    def __init__(
            self,
            n_units,
            n_layers=2,
            kernel_size=(3, 3),
            activation='relu',
            batch_norm=True
    ):
        super().__init__()

        modules = []
        for _ in range(n_layers):
            modules.append(
                torch.nn.Conv2d(
                    n_units,
                    n_units,
                    kernel_size=kernel_size,
                    padding=(kernel_size[0]//2, kernel_size[1]//2)
                )
            )
            if batch_norm:
                modules.append(
                    torch.nn.BatchNorm2d(
                        n_units
                    )
                )
            modules.append(
                ACTIVATION_DICT[activation]()
            )

        self.last_activation = modules.pop()

        self.block = torch.nn.Sequential(
            *modules
        )

    def forward(self, x):
        x = self.block(x) # + x
        return self.last_activation(x)


class ConvAE(torch.nn.Module, VAEBase):

    def __init__(
            self,
            input_shape,
            reconstruction_loss,
            prior=None,
            hidden_size=128,
            num_hidden=1,
            activation='relu',
            batch_norm=False
    ):
        super().__init__()

        activation_fn = ACTIVATION_DICT[activation]
        if prior is None:
            prior = NoPrior(latent_size=hidden_size)
        self.input_shape = input_shape
        self.prior = prior
        self.reconstruction = reconstruction_loss

        input = [
            torch.nn.Conv2d(
                input_shape[0],
                hidden_size,
                kernel_size=1
            )
        ]
        if batch_norm:
            input.append(
                torch.nn.BatchNorm2d(hidden_size)
            )
        input.append(activation_fn())

        self.input = torch.nn.Sequential(*input)

        self.block1 = ConvlBlock(
            hidden_size,
            n_layers=num_hidden,
            kernel_size=(3, 3),
            activation=activation,
            batch_norm=batch_norm
        )
        self.pool1 = torch.nn.AvgPool2d(2)

        self.block2 = ConvlBlock(
            hidden_size,
            n_layers=num_hidden,
            kernel_size=(3, 3),
            activation=activation,
            batch_norm=batch_norm
        )
        self.pool2 = torch.nn.AvgPool2d(2)

        self.block3 = ConvlBlock(
            hidden_size,
            n_layers=num_hidden,
            kernel_size=(3, 3),
            activation=activation,
            batch_norm=batch_norm
        )
        self.pool3 = torch.nn.AvgPool2d(2)

        self.pre_pool_size = hidden_size * input_shape[1] // 8 * input_shape[2] // 8

        self.pre_prior = torch.nn.Sequential(
            torch.nn.Linear(self.pre_pool_size, self.prior.input_size)
        )

        self.post_prior = torch.nn.Sequential(
            torch.nn.Linear(self.prior.latent_size, self.pre_pool_size),
            activation_fn()
        )

        self.block4 = ConvlBlock(
            hidden_size,
            n_layers=num_hidden,
            kernel_size=(3, 3),
            activation=activation,
            batch_norm=batch_norm
        )

        self.block5 = ConvlBlock(
            hidden_size,
            n_layers=num_hidden,
            kernel_size=(3, 3),
            activation=activation,
            batch_norm=batch_norm
        )

        self.block6 = ConvlBlock(
            hidden_size,
            n_layers=num_hidden,
            kernel_size=(3, 3),
            activation=activation,
            batch_norm=batch_norm
        )

        self.output = torch.nn.Conv2d(
            hidden_size,
            input_shape[0],
            kernel_size=1,
        )

        self.apply(init_weights)

    def forward(self, batch):
        batch = self.encode(batch)
        shape = batch['pre_codes'].shape
        batch['pre_codes'] = self.pre_prior(batch['pre_codes'].view(shape[0], -1))
        batch = self.prior(batch)
        batch['post_codes'] = self.post_prior(batch['codes']).view(shape)
        batch = self.decode(batch)
        return batch

    def encode(self, batch):
        x = batch['observations']
        x = self.input(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        batch['pre_codes'] = x
        # batch['pool_indices'] = [idx1, idx2, idx3]
        return batch

    def decode(self, batch):
        # x = torch.nn.functional.max_unpool2d(batch['post_codes'], batch['pool_indices'][2], self.pre_pool_size)
        # x = torch.nn.functional.max_unpool2d(self.block4(x), batch['pool_indices'][1], 2)
        # x = torch.nn.functional.max_unpool2d(self.block5(x), batch['pool_indices'][0], 2)

        x = torch.nn.functional.interpolate(batch['post_codes'], scale_factor=(2, 2))
        x = torch.nn.functional.interpolate(self.block4(x), scale_factor=(2, 2))
        x = torch.nn.functional.interpolate(self.block5(x), scale_factor=(2, 2))
        batch['pre_reconstructions'] = self.output(self.block6(x))
        batch = self.reconstruction(batch)
        return batch

# from dcase2020_task2.losses import MSEReconstruction
# ae = ConvAE((1, 128, 16), MSEReconstruction((1, 128, 16)))
# batch = {
#    'observations': torch.zeros((512, 1, 128, 16))
# }
# batch = ae(batch)
# print(batch)
