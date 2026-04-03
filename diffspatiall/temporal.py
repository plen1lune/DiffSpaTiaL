"""STL temporal operators (for-loop). Always, Eventually, Until, Next."""
import torch
import torch.nn as nn
from typing import Optional, List
from diffspatiall.utils import smooth_max, smooth_min


class Always(nn.Module):
    """G[a,b] — sliding-window min."""
    def __init__(self, interval=None):
        super().__init__()
        self.interval = interval

    def forward(self, signal, tau=1e-2):
        squeeze = signal.dim() == 3 and signal.shape[-1] == 1
        if squeeze:
            signal = signal.squeeze(-1)
        B, T = signal.shape
        a = self.interval[0] if self.interval else 0
        b = self.interval[-1] if self.interval else T - 1
        result = torch.empty_like(signal)
        for t in range(T):
            lo, hi = t + a, min(t + b, T - 1)
            if lo > T - 1:
                result[:, t] = 1e9
            else:
                result[:, t] = smooth_min(signal[:, lo:hi+1], dim=-1, tau=tau)
        return result.unsqueeze(-1) if squeeze else result


class Eventually(nn.Module):
    """F[a,b] — sliding-window max."""
    def __init__(self, interval=None):
        super().__init__()
        self.interval = interval

    def forward(self, signal, tau=1e-2):
        squeeze = signal.dim() == 3 and signal.shape[-1] == 1
        if squeeze:
            signal = signal.squeeze(-1)
        B, T = signal.shape
        a = self.interval[0] if self.interval else 0
        b = self.interval[-1] if self.interval else T - 1
        result = torch.empty_like(signal)
        for t in range(T):
            lo, hi = t + a, min(t + b, T - 1)
            if lo > T - 1:
                result[:, t] = -1e9
            else:
                result[:, t] = smooth_max(signal[:, lo:hi+1], dim=-1, tau=tau)
        return result.unsqueeze(-1) if squeeze else result


class Until(nn.Module):
    """phi1 U[a,b] phi2."""
    def __init__(self, interval=None):
        super().__init__()
        self.interval = interval

    def forward(self, signal1, signal2, tau=1e-2):
        squeeze = signal1.dim() == 3 and signal1.shape[-1] == 1
        if squeeze:
            signal1, signal2 = signal1.squeeze(-1), signal2.squeeze(-1)
        B, T = signal1.shape
        a = self.interval[0] if self.interval else 0
        b = self.interval[-1] if self.interval else T - 1

        result = torch.full((B, T), -1e9, device=signal1.device, dtype=signal1.dtype)
        for t in range(T):
            candidates = []
            for k in range(a, b + 1):
                if t + k >= T:
                    continue
                phi2_val = signal2[:, t + k]
                phi1_min = smooth_min(signal1[:, t:t+k+1], dim=-1, tau=tau)
                candidates.append(torch.min(phi2_val, phi1_min))
            if candidates:
                result[:, t] = smooth_max(torch.stack(candidates, dim=-1), dim=-1, tau=tau)
        return result.unsqueeze(-1) if squeeze else result


class Next(nn.Module):
    """X — shift signal by one timestep."""
    def forward(self, signal, **kwargs):
        squeeze = signal.dim() == 3 and signal.shape[-1] == 1
        if squeeze:
            signal = signal.squeeze(-1)
        B, T = signal.shape
        pad = torch.full((B, 1), -1e9, device=signal.device, dtype=signal.dtype)
        result = torch.cat([signal[:, 1:], pad], dim=1)
        return result.unsqueeze(-1) if squeeze else result


# boolean connectives on robustness signals

def stl_and(*signals):
    return torch.stack(signals, dim=-1).min(dim=-1).values

def stl_or(*signals):
    return torch.stack(signals, dim=-1).max(dim=-1).values

def stl_not(signal):
    return -signal
