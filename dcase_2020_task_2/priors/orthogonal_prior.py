from priors import PriorBase
import torch.nn
import numpy as np
from torch.nn import functional as F
import priors.utils


class OrthogonalPrior(PriorBase):

    def __init__(
            self,
            num_epochs=200,
            min_anneal=0,
            max_anneal=1,
            weight=1.0,
            latent_size=32,
            c_max=0.0,
            c_stop_epoch=-1,
            **kwargs
    ):
        super().__init__(weight=weight, c_max=c_max, c_stop_epoch=c_stop_epoch)
        self.latent_size_ = latent_size

        basis = np.concatenate(
            [
                -np.eye(latent_size, dtype=np.float32),
                np.eye(latent_size, dtype=np.float32)
            ]
        )
        self.slices = priors.utils.get_slices(basis, num_epochs, min_anneal=min_anneal, max_anneal=max_anneal)

    def forward(self, batch):
        batch['mus'] = batch['pre_codes'][:, :self.latent_size_]
        batch['logvars'] = batch['pre_codes'][:, self.latent_size_:]
        batch['stds'] = torch.exp(0.5 * batch['logvars'])
        batch['eps'] = torch.randn_like(batch['stds'])
        batch['codes'] = batch['mus'] + batch['eps'] * batch['stds']
        return batch

    def loss(self, batch):
        means = torch.from_numpy(self.slices[batch['epoch']]).to(batch['observations'].device)

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