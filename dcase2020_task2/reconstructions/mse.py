from reconstructions import ReconstructionBase
import torch
import torch.nn.functional as F


class MSE(ReconstructionBase):

    def __init__(self, weight=1.0, p=0.1, **kwargs):
        super().__init__(weight=weight)

        self.p = p

    def loss(self, batch, batch_abnormal=None):

        if batch_abnormal:
            normal_scores = batch['scores']
            abnormal_scores = batch_abnormal['scores']

            with torch.no_grad():
                phi = torch.kthvalue(normal_scores, int((1 - self.p) * normal_scores.shape[0]))[0]
            batch['tpr'] = torch.sigmoid(abnormal_scores - phi).mean()
            batch['fpr'] = torch.sigmoid(normal_scores - phi).mean()

            batch['reconstruction_loss'] = self.weight * normal_scores.mean() - abnormal_scores.mean()
            batch['tpr'] = batch['tpr'].item()
            batch['fpr'] = batch['fpr'].item()
        else:
            bce = F.mse_loss(batch['reconstructions'], batch['observations'], reduction='sum')
            batch['reconstruction_loss'] = self.weight * bce / (batch['observations'].shape[0])

        return batch['reconstruction_loss']

    def forward(self, batch):

        batch['visualizations'] = batch['pre_reconstructions']
        batch['reconstructions'] = batch['pre_reconstructions']
        batch['scores'] = (batch['reconstructions'] - batch['observations']).pow(2).mean(axis=(1, 2, 3))
        return batch
