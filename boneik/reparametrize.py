from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F

PI = np.pi
PI2 = 2 * PI


class NonPeriodicAngleReparametrization:
    """Reparametrize an contrained angle for unconstrained optimization.

    Combines a sigmoid and affine transformation to reparametrize
    the angle in such a way that it permits unconstrained optimization.
    This is code is adapted from `torch.distributions.constraints`.

    Deprecated, Use PeriodicAngleReparametrization instead.
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


class PeriodicAngleReparametrization:
    """Reparametrize an contrained angle for unconstrained optimization.

    Combines an affine transformation and complex number representation
    of angles.
    """

    def __init__(self, interval: Tuple[float, float] = (-np.pi, np.pi)) -> None:
        self.low = interval[0]
        self.scale = interval[1] - interval[0]

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        f = torch.angle(torch.view_as_complex(F.hardtanh(z)))  # [-pi..pi]
        n = (f + PI) / PI2  # [0..1]
        angle = n * self.scale + self.low  # [low..high]
        return angle

    def inv(self, angle: torch.Tensor) -> torch.Tensor:
        n = (angle - self.low) / self.scale  # [0..1]
        f = n * PI2 - PI  # [-pi..pi]
        return torch.stack((torch.cos(f), torch.sin(f)), -1)
