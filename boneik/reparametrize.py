from typing import Tuple
import numpy as np
import torch


class AngleReparametrization:
    """Reparametrize an contrained angle for unconstrained optimization.

    Combines a sigmoid and affine transformation to reparametrize
    the angle in such a way that it permits unconstrained optimization.
    This is code is adapted from `torch.distributions.constraints`.
    """

    def __init__(self, interval: Tuple[float, float] = (-np.pi, np.pi)) -> None:
        self.loc = interval[0]
        self.scale = interval[1] - interval[0]

    def __call__(self, angle: torch.Tensor) -> torch.Tensor:
        finfo = torch.finfo(angle.dtype)
        zeroone = torch.clamp(torch.sigmoid(angle), min=finfo.tiny, max=1.0 - finfo.eps)
        return self.loc + self.scale * zeroone

    def inv(self, uangle: torch.Tensor) -> torch.Tensor:
        zeroone = (uangle - self.loc) / self.scale
        finfo = torch.finfo(zeroone.dtype)
        zeroone = zeroone.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        return zeroone.log() - (-zeroone).log1p()
