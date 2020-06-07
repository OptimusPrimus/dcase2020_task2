from dcase2020_task2.losses import BaseReconstruction, BaseLoss
import torch
import torch.nn.functional as F


class AUC(BaseLoss):

    def __init__(self, weight=1.0, **kwargs):
        super().__init__()
        self.weight = weight

    def forward(self, batch_normal):

        assert batch_normal.get('scores') is not None, "cannot compute loss without scores"

        normal_scores = batch_normal['scores'][batch_normal['abnormal'] == 0]
        abnormal_scores = batch_normal['scores'][batch_normal['abnormal'] == 1]

        tprs = torch.sigmoid(abnormal_scores[:, None] - normal_scores[None, :]).mean(dim=0)
        batch_normal['tpr'] = tprs.mean()
        batch_normal['fpr'] = 0.5

        batch_normal['loss_raw'] = - batch_normal['tpr']
        batch_normal['loss'] = self.weight * batch_normal['loss_raw']

        # log some stuff...
        batch_normal['normal_scores_mean'] = normal_scores.mean()
        batch_normal['normal_scores_std'] = normal_scores.std()
        batch_normal['abnormal_scores_mean'] = abnormal_scores.mean()
        batch_normal['abnormal_scores_std'] = abnormal_scores.std()

        return batch_normal
