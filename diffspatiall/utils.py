"""LogSumExp-based smooth min/max."""
import torch


def smooth_max(x, dim=-1, tau=1e-2):
    return tau * torch.logsumexp(x / tau, dim=dim)


def smooth_min(x, dim=-1, tau=1e-2):
    return -smooth_max(-x, dim=dim, tau=tau)
