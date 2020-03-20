from priors import PriorBase
import torch.nn


class StandardNormalPrior(PriorBase):

    def __init__(
            self,
            weight=1.0,
            latent_size=32,
            c_max=0.0,
            c_stop_epoch=-1,
            **kwargs
    ):
        super().__init__(weight=weight, c_max=c_max, c_stop_epoch=c_stop_epoch)
        self.latent_size_ = latent_size

    def forward(self, batch):
        batch['mus'] = batch['pre_codes'][:, :self.latent_size_]
        batch['logvars'] = batch['pre_codes'][:, self.latent_size_:]
        batch['stds'] = torch.exp(0.5 * batch['logvars'])
        batch['eps'] = torch.randn_like(batch['stds'])
        batch['codes'] = batch['mus'] + batch['eps'] * batch['stds']
        return batch

    def loss(self, batch):
        batch['klds'] = -0.5 * (1 + batch['logvars'] - batch['mus'].pow(2) - batch['logvars'].exp())
        batch['prior_loss'] = batch['klds'].sum(1).mean(0)
        return self.weight_anneal(batch)

    @property
    def latent_size(self):
        return self.latent_size_

    @property
    def input_size(self):
        return self.latent_size_ * 2
