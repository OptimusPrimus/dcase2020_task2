from dcase2020_task2.losses import BaseReconstruction

import torch
import torch.nn.functional as F
from torch import nn as nn, distributions as D
import numpy as np


class NLLReconstruction(BaseReconstruction):

    def __init__(self, input_shape, weight=1.0, **kwargs):
        super().__init__()
        self.weight = weight
        self.input_shape = input_shape
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(np.prod(input_shape)))
        self.register_buffer('base_dist_var', torch.ones(np.prod(input_shape)))

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)


    def forward(self, batch):

        # prepare observations and prediction based on loss type:
        # use linear outputs & normalized observations as is
        # MAF eq 4 -- return mean and log std


        # log probability
        batch['log_proba'] = torch.sum(self.base_dist.log_prob(batch['u']) + batch['log_abs_det_jacobian'], dim=1)

        # scores
        batch['scores'] = - batch['log_proba']

        batch['visualizations'] = batch['reconstruction']

        # loss
        batch['reconstruction_loss_raw'] = - batch['log_proba'].mean()
        batch['reconstruction_loss'] = self.weight * batch['reconstruction_loss_raw']

        return batch
