from losses import ReconstructionBase
import torch
import torch.nn.functional as F


class BCE(ReconstructionBase):

    def __init__(self, weight=1.0, rho=0.2, **kwargs):
        super().__init__(weight=weight)
        self.rho = rho

    def loss(self, batch_normal, batch_abnormal, *args, **kwargs):
        normal_scores = batch_normal['scores']
        abnormal_scores = batch_abnormal['scores']

        batch_normal['normal_scores'] = normal_scores.mean()
        batch_normal['abnormal_scores'] = abnormal_scores.mean()

        abnormal_loss = torch.nn.functional.binary_cross_entropy_with_logits(abnormal_scores, torch.ones_like(abnormal_scores).to(abnormal_scores.device))
        normal_loss = torch.nn.functional.binary_cross_entropy_with_logits(normal_scores, torch.zeros_like(normal_scores).to(normal_scores.device))

        batch_normal['reconstruction_loss'] = self.weight * (abnormal_loss + normal_loss)

        return batch_normal['reconstruction_loss']

    def forward(self, batch):
        batch['visualizations'] = batch['pre_reconstructions']
        batch['losses'] = batch['pre_reconstructions']
        batch['scores'] = (batch['losses'] - batch['observations']).pow(2).mean(axis=(1, 2, 3))
        return batch
