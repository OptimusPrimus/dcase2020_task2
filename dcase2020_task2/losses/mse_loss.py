from losses import ReconstructionBase
import torch.nn.functional as F


class MSE(ReconstructionBase):

    def __init__(self, weight=1.0, p=0.1, **kwargs):
        super().__init__(weight=weight)

        self.p = p

    def loss(self, batch, *args, **kwargs):
        bce = F.mse_loss(batch['losses'], batch['observations'], reduction='mean')
        batch['reconstruction_loss'] = self.weight * bce # / (batch['observations'].shape[0])
        return batch['reconstruction_loss']

    def forward(self, batch):

        batch['visualizations'] = batch['pre_reconstructions']
        batch['losses'] = batch['pre_reconstructions']
        batch['scores'] = (batch['losses'] - batch['observations']).pow(2).mean(axis=(1, 2, 3))
        return batch
