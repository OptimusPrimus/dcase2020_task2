from dcase2020_task2.losses import BaseLoss
import torch


class NP(BaseLoss):

    def __init__(self, weight=1.0, rho=0.2, **kwargs):
        super().__init__()
        self.weight = weight
        self.rho=0.2

    def forward(self, batch_normal, batch_abnormal, *args, **kwargs):

        assert batch_normal.get('scores'), "cannot compute loss without scores"
        assert batch_abnormal.get('scores'), "cannot compute loss without scores"

        normal_scores = batch_normal['scores']
        abnormal_scores = batch_abnormal['scores']

        with torch.no_grad():
            phi = torch.kthvalue(normal_scores, int((1 - self.rho) * normal_scores.shape[0]))[0]

        batch_normal['tpr'] = torch.sigmoid(abnormal_scores - phi).mean()
        batch_normal['fpr'] = torch.sigmoid(normal_scores - phi).mean()

        batch_normal['loss_raw'] = batch_normal['fpr'] - batch_normal['tpr']
        batch_normal['loss'] = self.weight * batch_normal['loss_raw']

        # log some stuff...
        batch_normal['normal_scores_mean'] = normal_scores.mean()
        batch_normal['normal_scores_std'] = normal_scores.std()
        batch_normal['abnormal_scores_mean'] = abnormal_scores.mean()
        batch_normal['abnormal_scores_std'] = normal_scores.std()

        return batch_normal
