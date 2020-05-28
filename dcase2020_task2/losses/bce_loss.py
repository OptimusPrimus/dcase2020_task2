from dcase2020_task2.losses import BaseReconstruction, BaseLoss
import torch
import torch.nn.functional as F


class BCE(BaseLoss):

    def __init__(self, weight=1.0, **kwargs):

        super().__init__()
        self.weight = weight

    def forward(self, batch_normal, batch_abnormal):

        assert batch_normal.get('scores'), "cannot compute loss without scores"
        assert batch_abnormal.get('scores'), "cannot compute loss without scores"

        normal_scores = batch_normal['scores']
        abnormal_scores = batch_abnormal['scores']

        abnormal_loss = F.binary_cross_entropy_with_logits(
            abnormal_scores,
            torch.ones_like(abnormal_scores).to(abnormal_scores.device)
        )

        normal_loss = F.binary_cross_entropy_with_logits(
            normal_scores,
            torch.zeros_like(normal_scores).to(normal_scores.device)
        )

        batch_normal['loss_raw'] = 0.5 * (abnormal_loss + normal_loss)
        batch_normal['loss'] = self.weight * batch_normal['loss_raw']

        # log some stuff...
        batch_normal['normal_scores_mean'] = normal_scores.mean()
        batch_normal['normal_scores_std'] = normal_scores.std()
        batch_normal['abnormal_scores_mean'] = abnormal_scores.mean()
        batch_normal['abnormal_scores_std'] = normal_scores.std()

        return batch_normal
