from dcase2020_task2.priors import PriorBase
import torch.nn


class NoPrior(PriorBase):

    def __init__(self, latent_size=256, **kwargs):
        super().__init__()
        self.latent_size_ = latent_size

    def forward(self, batch):
        """ No modification """
        batch['codes'] = batch['pre_codes']
        return batch

    def loss(self, batch):
        """ Zero Loss """
        batch['prior_loss'] = torch.tensor(0)
        return batch['prior_loss']

    @property
    def latent_size(self):
        return self.latent_size_

    @property
    def input_size(self):
        return self.latent_size_
