"""
Fabrizio Ventola
TU Darmstadt
"""

from priors import PriorBase
import torch.nn
import math

"""
Most of this code is taken from:
https://github.com/AntixK/PyTorch-VAE
by Anand Krishnamoorthy
"""

class BetaTCVaePrior(PriorBase):
    num_iter = 0

    def __init__(
            self,
            dataset_size,
            weight=1.0,
            latent_size=32,
            c_max=0.0,
            c_stop_epoch=-1,
            anneal_steps=200,
            alpha=1.,
            beta=6.,
            gamma=1.,
            **kwargs
    ):
        super().__init__(weight=weight, c_max=c_max, c_stop_epoch=c_stop_epoch)
        self.latent_size_ = latent_size
        self.dataset_size = dataset_size
        self.anneal_steps = anneal_steps
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, batch):
        batch['mus'] = batch['pre_codes'][:, :self.latent_size_]
        batch['logvars'] = batch['pre_codes'][:, self.latent_size_:]
        batch['stds'] = torch.exp(0.5 * batch['logvars'])
        batch['eps'] = torch.randn_like(batch['stds'])
        batch['codes'] = batch['mus'] + batch['eps'] * batch['stds']
        return batch

    def loss(self, batch):
        """
         alpha * MI_LOSS +
         w * (
               beta * TC_LOSS +
               anneal_r * gamma * KLD_LOSS
             )
        """

        dataset_size = self.dataset_size
        log_q_zx = self.log_density_gaussian(batch['codes'], batch['mus'], batch['logvars']).sum(dim=1)

        zeros = torch.zeros_like(batch['codes'])
        log_p_z = self.log_density_gaussian(batch['codes'], zeros, zeros).sum(dim = 1)

        batch_size, latent_dim = batch['codes'].shape
        mat_log_q_z = self.log_density_gaussian(batch['codes'].view(batch_size, 1, latent_dim),
                                                batch['mus'].view(1, batch_size, latent_dim),
                                                batch['logvars'].view(1, batch_size, latent_dim))

        strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))
        importance_weights = torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size -1)).to(batch['mus'].device)
        importance_weights.view(-1)[::batch_size] = 1 / dataset_size
        importance_weights.view(-1)[1::batch_size] = strat_weight
        importance_weights[batch_size - 2, 0] = strat_weight
        log_importance_weights = importance_weights.log()

        mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)

        log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
        log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

        mi_loss  = (log_q_zx - log_q_z).mean()
        tc_loss = (log_q_z - log_prod_q_z).mean()
        kld_loss = (log_prod_q_z - log_p_z).mean()

        if self.training:
            self.num_iter += 1
            anneal_rate = min(0 + 1 * self.num_iter / self.anneal_steps, 1)
        else:
            anneal_rate = 1.

        loss = self.alpha * mi_loss + \
               self.weight * (self.beta * tc_loss +
                         anneal_rate * self.gamma * kld_loss)
        batch['prior_loss'] = loss

        return batch['prior_loss'] # TODO revise annealing

    @staticmethod
    def log_density_gaussian(x, mu, logvar):
        """
        Computes the log pdf of the Gaussian with parameters mu and logvar at x
        :param x: (Tensor) Point at which Gaussian PDF is to be evaluated
        :param mu: (Tensor) Mean of the Gaussian distribution
        :param logvar: (Tensor) Log variance of the Gaussian distribution
        :return: log_density: (Tensor) the log pdf at x
        """
        norm = - 0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density

    @property
    def latent_size(self):
        return self.latent_size_

    @property
    def input_size(self):
        return self.latent_size_ * 2
