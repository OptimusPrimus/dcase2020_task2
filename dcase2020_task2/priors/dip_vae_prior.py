"""
Fabrizio Ventola
TU Darmstadt
"""

from priors import PriorBase
import torch.nn

"""
Most of this code is taken from:
https://github.com/AntixK/PyTorch-VAE
by Anand Krishnamoorthy
"""

class DIPVaePrior(PriorBase):

    def __init__(
            self,
            weight=1.0,
            latent_size=32,
            c_max=0.0,
            c_stop_epoch=-1,
            lambda_diag=10.,
            lambda_offdiag=5.,
            dip_first=False,
            **kwargs
    ):
        super().__init__(weight=weight, c_max=c_max, c_stop_epoch=c_stop_epoch)
        self.latent_size_ = latent_size
        self.lambda_diag = lambda_diag
        self.lambda_offdiag = lambda_offdiag
        self.dip_first = dip_first

    def forward(self, batch):
        batch['mus'] = batch['pre_codes'][:, :self.latent_size_]
        batch['logvars'] = batch['pre_codes'][:, self.latent_size_:]
        batch['stds'] = torch.exp(0.5 * batch['logvars'])
        batch['eps'] = torch.randn_like(batch['stds'])
        batch['codes'] = batch['mus'] + batch['eps'] * batch['stds']
        return batch

    def loss(self, batch):
        mu = batch['mus']
        log_var = batch['logvars']

        kld_weight = 1 #* kwargs['M_N'] # Account for the minibatch samples from the dataset

        kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # DIP Loss
        centered_mu = mu - mu.mean(dim=1, keepdim = True) # [B x D]
        cov_mu = centered_mu.t().matmul(centered_mu).squeeze() # [D X D]

        if self.dip_first:
            # For DIp Loss I
            cov_z = cov_mu
            print('dip_I')
        else:
            # Add Variance for DIP Loss II
            cov_z = cov_mu + torch.mean(torch.diagonal((2. * log_var).exp(), dim1 = 0), dim = 0) # [D x D]
            print('dip_II')

        cov_diag = torch.diag(cov_z) # [D]
        cov_offdiag = cov_z - torch.diag(cov_diag) # [D x D]
        dip_loss = self.lambda_offdiag * torch.sum(cov_offdiag ** 2) + \
                   self.lambda_diag * torch.sum((cov_diag - 1) ** 2)

        loss = kld_weight * kld_loss + dip_loss # TODO double check kld_loss sign

        batch['prior_loss'] = loss
        return batch['prior_loss'] # TODO revise annealing

    @property
    def latent_size(self):
        return self.latent_size_

    @property
    def input_size(self):
        return self.latent_size_ * 2
