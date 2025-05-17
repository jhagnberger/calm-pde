import torch


class RelL2Loss():
    """Relative L2 loss for PDEs adopted from https://github.com/BaratiLab/FactFormer/blob/main/loss_fn.py"""

    def __init__(self, dim=-2, eps=1e-5, reduction='sum', reduce_all=True):
        self.dim = dim
        self.eps = eps
        self.reduction = reduction
        self.reduce_all = reduce_all
    
    def __call__(self, y_hat, y):
        # y is the ground truth
        # first reduce wrt to grid point dimension
        # i.e. mesh weighted
        # return torch.mean(
        #         (x - y) ** 2 / (y ** 2 + 1e-6), dim=(-1, -2)).sqrt().mean()
        assert y_hat.shape == y.shape

        reduce_fn = torch.mean if self.reduction == 'mean' else torch.sum

        y_norm = reduce_fn((y ** 2), dim=self.dim)
        mask = y_norm < self.eps
        y_norm[mask] = self.eps
        diff = reduce_fn((y_hat - y) ** 2, dim=self.dim)
        diff = diff / y_norm  # [b, c]
        if self.reduce_all:
            diff = diff.sqrt().mean()    # mean across channels and batch and any other dimensions
        else:
            diff = diff.sqrt()  # do nothing
        return diff
