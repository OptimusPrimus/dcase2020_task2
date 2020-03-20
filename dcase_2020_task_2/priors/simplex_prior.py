from priors import PriorBase
import torch.nn
import numpy as np
from torch.nn import functional as F
import priors.utils


class SimplexPrior(PriorBase):

    def __init__(
            self,
            anneal_stop_epoch=200,
            min_anneal=0,
            max_anneal=1,
            weight=1.0,
            latent_size=32,
            c_max=0.0,
            c_stop_epoch=-1,
            **kwargs
    ):
        super().__init__(weight=weight, c_max=c_max, c_stop_epoch=c_stop_epoch)
        self.anneal_stop_epoch = anneal_stop_epoch
        self.latent_size_ = latent_size

        basis = self.simplex_coordinates(latent_size)
        self.slices = priors.utils.get_slices(basis, anneal_stop_epoch, min_anneal=min_anneal, max_anneal=max_anneal)

    def forward(self, batch):
        batch['mus'] = batch['pre_codes'][:, :self.latent_size]
        batch['logvars'] = batch['pre_codes'][:, self.latent_size_:]
        batch['stds'] = torch.exp(0.5 * batch['logvars'])
        batch['eps'] = torch.randn_like(batch['stds'])
        batch['codes'] = batch['mus'] + batch['eps'] * batch['stds']
        return batch

    def loss(self, batch):
        if batch['epoch'] < self.anneal_stop_epoch:
            means = torch.from_numpy(self.slices[batch['epoch']]).to(batch['observations'].device)
        else:
            means = torch.from_numpy(self.slices[-1]).to(batch['observations'].device)

        klds = []
        for mean in means:
            klds.append(0.5 * (- batch['logvars'] - 1 + batch['logvars'].exp() + (batch['mus'] - mean[None, :]).pow(2)))

        klds = torch.stack(klds)
        gate = F.softmax(-klds, dim=0)
        gated_klds = gate * klds
        batch['prior_loss'] = torch.mean(gated_klds, 0).sum(1).mean(0, True)
        return self.weight_anneal(batch)

    @property
    def latent_size(self):
        return self.latent_size_

    @property
    def input_size(self):
        return self.latent_size_ * 2

    @staticmethod
    def simplex_coordinates(num_dimensions):
        # This function is adopted from the Simplex Coordinates library
        # https://people.sc.fsu.edu/~jburkardt/py_src/simplex_coordinates/simplex_coordinates.html
        """
        This function computes the cartesian coordinates of a m-dimensional simplex
        :param num_dimensions: dimensionality of the simplex
        :return: the simplex coordinates
        """
        x = np.zeros([num_dimensions, num_dimensions + 1])

        for j in range(0, num_dimensions):
            x[j, j] = 1.0

        a = (1.0 - np.sqrt(1.0 + num_dimensions)) / num_dimensions
        x[:, num_dimensions] = a

        c = x.sum(axis=1) / (num_dimensions + 1)
        x = x - c[:, None]

        n = 0
        for s in x[:, 0] ** 2:
            n = n + s
            n = np.sqrt(n)
        return (x / n).T
