from losses import ReconstructionBase
import torch.nn.functional as F


class MSE(ReconstructionBase):

    def __init__(self, weight=1.0, p=0.1, **kwargs):
        super().__init__(weight=weight)

        self.p = p

    def loss(self, batch, *args, **kwargs):
        bce = F.mse_loss(batch['predictions'], batch['observations'], reduction='sum')
        batch['reconstruction_loss'] = self.weight * bce
        return batch['reconstruction_loss']

    def forward(self, batch):

        batch['visualizations'] = batch['pre_reconstructions']
        batch['predictions'] = batch['pre_reconstructions']
        batch['scores'] = (batch['predictions'] - batch['observations']).pow(2).mean(axis=(1, 2, 3))
        return batch
