from reconstructions import ReconstructionBase
import torch
import torch.nn.functional as F


class CategoricalCrossEntropy(ReconstructionBase):

    def __init__(self, weight=1.0, input_shape=(1, 64, 64), **kwargs):
        super().__init__(weight=weight)
        self.input_shape = input_shape
        self.out = torch.nn.Conv2d(input_shape[0], 256 * input_shape[0], 1)

    def loss(self, batch):
        recon = torch.transpose(batch['expanded'], 2, 4).reshape(-1, 256)
        origi = torch.transpose((batch['observations'] * 255).long()[:, :, None], 2, 4).reshape(-1)
        bce = F.cross_entropy(recon, origi, reduction='sum')
        batch['reconstruction_loss'] = self.weight * bce / (batch['observations'].shape[0])
        return batch['reconstruction_loss']

    def forward(self, batch):
        batch['expanded'] = self.out(batch['pre_reconstructions']).view(
            -1,
            self.input_shape[0],
            256,
            self.input_shape[1],
            self.input_shape[2]
        )
        batch['reconstructions'] = torch.argmax(batch['expanded'], dim=2).float() / 255
        batch['visualizations'] = batch['reconstructions']
        return batch
