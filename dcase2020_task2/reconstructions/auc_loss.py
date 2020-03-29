from reconstructions import ReconstructionBase
import torch
import torch.nn.functional as F


class AUC(ReconstructionBase):

    def __init__(self, weight=1.0, rho=0.2, mse_weight=0.0, **kwargs):
        super().__init__(weight=weight)
        self.rho = rho
        self.mse_weight = mse

    def loss(self, batch_normal, batch_abnormal, *args, **kwargs):
        normal_scores = batch_normal['scores']
        abnormal_scores = batch_abnormal['scores']

        batch_normal['normal_scores'] = normal_scores.mean()
        batch_normal['abnormal_scores'] = abnormal_scores.mean()

        batch_normal['mse_normal'] = F.mse_loss(batch_normal['reconstructions'], batch_normal['observations'])
        batch_normal['mse_abnormal'] = F.mse_loss(batch_abnormal['reconstructions'], batch_abnormal['observations'])    # batch_abnormal['observations'])

        tprs = torch.sigmoid((abnormal_scores[:, None] - normal_scores[None, :])).mean(dim=0)
        batch_normal['tpr'] = tprs.mean()
        batch_normal['fpr'] = 0.5

        batch_normal['reconstruction_loss'] = self.weight*(-batch_normal['tpr']+self.mse*batch_normal['mse_normal'])

        return batch_normal['reconstruction_loss']

    def forward(self, batch):
        batch['visualizations'] = batch['pre_reconstructions']
        batch['reconstructions'] = batch['pre_reconstructions']
        batch['scores'] = (batch['reconstructions'] - batch['observations']).pow(2).mean(axis=(1, 2, 3))
        return batch
