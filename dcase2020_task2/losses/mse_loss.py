from dcase2020_task2.losses import BaseReconstruction
import torch.nn.functional as F


class MSEReconstruction(BaseReconstruction):

    def __init__(self, input_shape, weight=1.0, size_average=True, **kwargs):
        super().__init__()
        self.input_shape = input_shape
        self.weight = weight
        self.size_average = size_average

    def forward(self, batch):

        # TODO: maybe clipping for visualization
        batch['visualizations'] = batch['pre_reconstructions']

        # prepare observations and prediction based on loss type:
        # use linear outputs & normalized observations as is
        batch['predictions'] = batch['pre_reconstructions']

        # scores
        batch['scores'] = F.mse_loss(
            batch['predictions'],
            batch['observations'],
            reduction='none'
        ).reshape(len(batch['predictions']), -1).mean(1)

        # loss
        batch['reconstruction_loss_raw'] = batch[f'scores'].mean() if self.size_average else batch[f'scores'].sum()
        batch['reconstruction_loss'] = self.weight * batch['reconstruction_loss_raw']

        return batch
