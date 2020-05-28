from dcase2020_task2.losses import BaseReconstruction
import torch.nn.functional as F


class MSEReconstruction(BaseReconstruction):

    def __init__(self, weight=1.0, size_average=True, **kwargs):
        super().__init__()
        self.weight = weight
        self.size_average = size_average

    def forward(self, batch):

        # TODO: maybe clipping for visualization
        batch['visualizations'] = batch['pre_reconstructions']

        # prepare observations and prediction based on loss type:
        # use linear outputs
        batch['predictions'] = batch['pre_reconstructions']
        # use normalized observations as is
        batch['targets'] = batch['observations']

        # scores
        batch['scores'] = F.mse_loss(
            batch['predictions'],
            batch['targets'],
            reduction='none'
        ).view(len(batch), -1).mean(1)
        assert len(batch) == len(batch['targets'])

        # loss
        batch['reconstruction_loss_raw'] = batch[f'scores'].mean() if self.size_average else batch[f'scores'].sum()
        batch['reconstruction_loss'] = self.weight * batch['reconstruction_loss_raw']

        return batch
