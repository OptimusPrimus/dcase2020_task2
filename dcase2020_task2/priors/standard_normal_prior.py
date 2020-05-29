from dcase2020_task2.priors import PriorBase
import torch.nn

class StandardNormalPrior(PriorBase):

    def __init__(
            self,
            weight=1.0,
            latent_size=32,
            **kwargs
    ):
        super().__init__()
        self.weight = weight
        self.latent_size_ = latent_size

    def forward(self, batch):
        """ Sample a batch from the approximate posterior """
        batch['mus'] = batch['pre_codes'][:, :self.latent_size_]
        batch['logvars'] = batch['pre_codes'][:, self.latent_size_:]

        batch['stds'] = torch.exp(0.5 * batch['logvars'])
        batch['eps'] = torch.randn_like(batch['stds'])

        batch['codes'] = batch['mus'] + batch['eps'] * batch['stds']

        batch['prior_loss_raw'] = (-0.5 * (1 + batch['logvars'] - batch['mus'].pow(2) - batch['logvars'].exp())).sum()
        batch['prior_loss'] = self.weight * batch['prior_loss_raw']

        # log some stuff...
        batch['mean_mus'] = batch['mus'].mean()
        batch['std_mus'] = batch['mus'].std()
        batch['mean_logvar'] = batch['logvars'].mean()
        batch['std_logvar'] = batch['logvars'].std()

        return batch

    @property
    def latent_size(self):
        return self.latent_size_

    @property
    def input_size(self):
        return self.latent_size_ * 2
