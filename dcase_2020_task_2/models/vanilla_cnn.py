import torch.nn
from models import VAEBase
from models.layers import Convolution2dLayer, Deconvolution2dLayer, FullyConnectedLayer
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.init as init


class VanillaCNN(torch.nn.Module, VAEBase):

    def __init__(
            self,
            image_size,
            reconstruction_loss,
            prior,
            normalize_fun,
            conv_channels=[32, 32, 32],
            hidden_sizes=[256, 256]
    ):
        super().__init__()

        self.image_size = image_size[1:]
        image_channels = image_size[0]
        self.image_channels = image_channels
        image_size = self.image_size
        self.reconstruction = reconstruction_loss
        self.prior = prior

        self.normalize_fun = normalize_fun

        self.conv_channels = conv_channels
        latent_size = prior.latent_size
        self.latent_size = latent_size
        # last layer of encoder must have the size expected from prior
        self.hidden_sizes = hidden_sizes

        self.conv1 = Convolution2dLayer(
                padding=0,
                kernel_size=2,
                in_channels=image_channels,
                out_channels=conv_channels[0])
        size1 = self.conv1.out_size(image_size)

        self.conv2 = Convolution2dLayer(
                padding=0,
                kernel_size=3,
                in_channels=conv_channels[0],
                out_channels=conv_channels[1])
        size2 = self.conv2.out_size(size1)

        self.conv3 = Convolution2dLayer(
                padding=0,
                kernel_size=3,
                in_channels=conv_channels[1],
                out_channels=conv_channels[2])
        size3 = self.conv3.out_size(size2)
        self.size3 = size3
        self.num_conv_activations = size3[0] * size3[1] * conv_channels[2]

        self.encoder_hidden1 = FullyConnectedLayer(
                in_size=self.num_conv_activations,
                out_size=hidden_sizes[0])

        self.encoder_hidden2 = FullyConnectedLayer(
                in_size=hidden_sizes[0],
                out_size=hidden_sizes[1])

        self.encoder_hidden3 = FullyConnectedLayer(
                in_size=hidden_sizes[1],
                out_size=prior.input_size,
                activation=None
        )

        self.decoder_hidden_latent = FullyConnectedLayer(
                in_size=latent_size,
                out_size=hidden_sizes[1])
        self.decoder_hidden2 = FullyConnectedLayer(
                in_size=hidden_sizes[1],
                out_size=hidden_sizes[0])
        self.decoder_hidden1 = FullyConnectedLayer(
                in_size=hidden_sizes[0],
                out_size=self.num_conv_activations)

        self.deconv3 = Deconvolution2dLayer.matching_input_and_output_size(
                input_size=size3,
                output_size=size2,
                in_channels=conv_channels[2],
                out_channels=conv_channels[1],
                kernel_size=3,
                padding=0)

        self.deconv2 = Deconvolution2dLayer.matching_input_and_output_size(
                input_size=size2,
                output_size=size1,
                in_channels=conv_channels[1],
                out_channels=conv_channels[0],
                padding=0,
                kernel_size=3)

        # preactivate so we can tack on the sigmoid activation afterwards.
        self.deconv1 = Deconvolution2dLayer.matching_input_and_output_size(
                input_size=size1,
                output_size=self.image_size,
                in_channels=conv_channels[0],
                out_channels=image_channels,
                kernel_size=2,
                padding=0,
                preactivate=True)

    def forward(self, batch):
        batch = self.normalize_fun(batch)
        batch = self.encode(batch)
        batch = self.prior(batch)
        batch = self.decode(batch)
        return batch

    def encode(self, batch):
        x = self.conv1(batch['normalized_observations'])
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.num_conv_activations)
        x = self.encoder_hidden1(x)
        x = self.encoder_hidden2(x)
        batch['pre_codes'] = self.encoder_hidden3(x)
        return batch

    def decode(self, batch):
        x = self.decoder_hidden_latent(batch['codes'])
        x = self.decoder_hidden2(x)
        x = self.decoder_hidden1(x)
        x = x.view(-1, self.conv_channels[2], self.size3[0], self.size3[1])
        x = self.deconv3(x)
        x = self.deconv2(x)
        batch['pre_reconstructions'] = self.deconv1(x)
        batch = self.reconstruction(batch)
        return batch
