from reconstructions import ReconstructionBase
import torch


class NP(ReconstructionBase):

    def __init__(self, weight=1.0, rho=0.2, **kwargs):
        super().__init__(weight=weight)
        self.rho = rho

    def loss(self, batch_normal, batch_abnormal, *args, **kwargs):
        normal_scores = batch_normal['scores']
        abnormal_scores = batch_abnormal['scores']
        with torch.no_grad():
            phi = torch.kthvalue(normal_scores, int((1 - self.rho) * normal_scores.shape[0]))[0]
        batch_normal['tpr'] = torch.sigmoid(abnormal_scores - phi).mean()
        batch_normal['fpr'] = torch.sigmoid(normal_scores - phi).mean()
        batch_normal['reconstruction_loss'] = self.weight * (batch_normal['fpr'] - batch_normal['tpr'])

        return batch_normal['reconstruction_loss']

    def forward(self, batch):
        batch['visualizations'] = batch['pre_reconstructions']
        batch['reconstructions'] = batch['pre_reconstructions']
        batch['scores'] = (batch['reconstructions'] - batch['observations']).pow(2).sum(axis=(1, 2, 3))
        return batch
