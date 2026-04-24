import torch

class Oracle:
    """
    Base oracle interface.

    Any objective used with PBM should implement:
        f_batch(X): returns shape (J,)
        g_batch(X): returns shape (d, J)

    where X has shape (d, J).
    """
    def f_batch(self, X):
        raise NotImplementedError

    def g_batch(self, X):
        raise NotImplementedError


class QuadraticOracle(Oracle):
    """
    f(x) = 0.5 * x^T Q x
    """

    def __init__(self, Q):
        self.Q = Q

    def f_batch(self, X):
        return 0.5 * torch.sum(X * (self.Q @ X), dim=0)

    def g_batch(self, X):
        return self.Q @ X


class QuadraticL1Oracle(Oracle):
    """
    f(x) = 0.5 * x^T Q x + lambda * ||x||_1
    """

    def __init__(self, Q, lam=1e-2):
        self.Q = Q
        self.lam = lam

    def f_batch(self, X):
        quad = 0.5 * torch.sum(X * (self.Q @ X), dim=0)
        l1 = self.lam * torch.sum(torch.abs(X), dim=0)
        return quad + l1

    def g_batch(self, X):
        return self.Q @ X + self.lam * torch.sign(X)