import time
import torch


def time_gpu(fn):
    """
    Time a function in milliseconds, synchronizing CUDA if available.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()
    output = fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.time()
    return 1000.0 * (end - start), output


def make_psd_matrix(d, device=None, dtype=torch.float32, seed=0):
    """
    Create a random positive semidefinite matrix Q = A^T A.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    A = torch.randn(d, d, device=device, dtype=dtype)
    return A.T @ A