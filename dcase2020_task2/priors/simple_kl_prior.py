from priors import PriorBase
import torch.nn


class SimpleKLPrior(PriorBase):

    def __init__(self, weight=1.0, input_size=256, latent_size=32, c_max=0.0, c_stop_epoch=-1, **kwargs):
        super().__init__(weight=weight, c_max=c_max, c_stop_epoch=c_stop_epoch)
        self.latent_size_ = latent_size

    def forward(self, batch):
        batch['codes'] = batch['pre_codes']
        return batch

    def loss(self, batch):

        mean = torch.mean(batch['codes'], dim=0)
        var = torch.var(batch['codes'], dim=0)
        batch['prior_loss'] = 0.5 * (var.prod() + (1/var).sum() + (mean.pow(2)*(1/var)).sum() - mean.shape[0])

        return self.weight_anneal(batch)

    @property
    def latent_size(self):
        return self.latent_size_

    @property
    def input_size(self):
        return self.latent_size_
