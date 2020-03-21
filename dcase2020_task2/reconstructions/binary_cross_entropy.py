from reconstructions import ReconstructionBase
import torch
import torch.nn.functional as F


class BinaryCrossEntropy(ReconstructionBase):

    def __init__(self, weight=1.0, **kwargs):
        super().__init__(weight=weight)

    def loss(self, batch):
        bce = F.binary_cross_entropy(batch['probabilities'], batch['observations'], reduction='sum')
        batch['reconstruction_loss'] = self.weight * bce / (batch['observations'].shape[0])
        return batch['reconstruction_loss']

    def forward(self, batch):
        batch['probabilities'] = torch.sigmoid(batch['pre_reconstructions'])
        batch['visualizations'] = batch['probabilities']
        batch['reconstructions'] = batch['probabilities'] > 0.5
        return batch
