from priors import PriorBase
import torch.nn


class NoPrior(PriorBase):

    def __init__(self, weight=1.0, input_size=256, latent_size=32, c_max=0.0, c_stop_epoch=-1, **kwargs):
        super().__init__(weight=weight, c_max=c_max, c_stop_epoch=c_stop_epoch)
        self.latent_size_ = latent_size
        self.lin = torch.nn.Linear(input_size, latent_size)

    def forward(self, batch):
        batch['codes'] = self.lin(batch['pre_codes'])
        return batch

    def loss(self, batch):
        batch['prior_loss'] = torch.tensor(0)
        return self.weight_anneal(batch)

    @property
    def latent_size(self):
        return self.latent_size_

    @property
    def input_size(self):
        return self.latent_size_
