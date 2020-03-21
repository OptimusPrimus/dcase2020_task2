from reconstructions import ReconstructionBase
import torch
import torch.nn.functional as F


class MSE(ReconstructionBase):

    def __init__(self, weight=1.0, **kwargs):
        super().__init__(weight=weight)

    def loss(self, batch):
        bce = F.mse_loss(batch['reconstructions'], batch['observations'], reduction='sum')
        batch['reconstruction_loss'] = self.weight * bce / (batch['observations'].shape[0])
        return batch['reconstruction_loss']

    def forward(self, batch):

        batch['visualizations'] = batch['pre_reconstructions']
        batch['reconstructions'] = batch['pre_reconstructions']
        batch['predictions'] = (batch['reconstructions'] - batch['observations']).pow(2).mean(axis=(1, 2, 3))
        return batch
