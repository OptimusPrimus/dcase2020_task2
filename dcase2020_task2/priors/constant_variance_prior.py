from dcase2020_task2.priors import PriorBase
import torch.nn


class CVPrior(PriorBase):

    def __init__(self, latent_size=256, **kwargs):
        super().__init__()
        self.latent_size_ = latent_size

    def forward(self, batch):

        """ No modification """
        batch['codes'] = batch['pre_codes']

        batch['prior_loss'] = torch.norm(batch['pre_codes'], p=2, dim=1) ** 2

        # log some stuff...
        batch['std_prior_loss'] = batch['prior_loss'].std()

        batch['prior_loss'] = batch['prior_loss'].mean()

        return batch

    @property
    def latent_size(self):
        return self.latent_size_

    @property
    def input_size(self):
        return self.latent_size_
