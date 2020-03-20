from auxiliary_losses import AuxiliaryLossBase
import torch
import torch.nn.functional as F


class FactorVAE(AuxiliaryLossBase, torch.nn.Module):

    def __init__(self, model, weight=1.0):
        super().__init__(weight=weight)
        self.model = model
        self.prev_batch = None

    def forward(self, x):
        return self.model(x)

    def auxiliary_loss(self, batch):
        discriminator = self(batch['codes'])
        batch['auxiliary_loss'] = self.weight * (discriminator[:, :1] - discriminator[:, 1:]).mean()
        return batch['auxiliary_loss']

    def training_loss(self, batch):
        device = batch['codes'].device

        if self.prev_batch is not None:
            d_z_prim_prev = self(self.prev_batch['codes'])
            perm = permute_dims(batch['codes']).detach()
            d_z_prim_perm = self(perm)
            ones = torch.ones(d_z_prim_prev.shape[0], dtype=torch.long, device=device)
            zeros = torch.zeros(d_z_prim_perm.shape[0], dtype=torch.long, device=device)
            factor_loss = 0.5 * (
                        F.cross_entropy(d_z_prim_prev, zeros) + F.cross_entropy(d_z_prim_perm, ones)
            )
            batch['factor_loss'] = factor_loss
        else:
            factor_loss = torch.tensor(0.0, requires_grad=True, device=device)
            batch['factor_loss'] = factor_loss

        self.prev_batch = batch
        return batch['factor_loss']


def permute_dims(z):
    assert z.dim() == 2
    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)
