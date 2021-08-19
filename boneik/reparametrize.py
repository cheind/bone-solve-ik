from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F

PI = np.pi
PI2 = 2 * PI


class NonPeriodicAngleReparametrization:
    """Reparametrize an contrained angle for unconstrained optimization."""

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
        f = torch.atan2(z[:, 1], z[:, 0])
        # f = torch.angle(torch.view_as_complex(F.hardtanh(z)))  # [-pi..pi]
        n = (f + PI) / PI2  # [0..1]
        angle = n * self.scale + self.low  # [low..high]
        return angle

    def inv(self, angle: torch.Tensor) -> torch.Tensor:
        n = (angle - self.low) / self.scale  # [0..1]
        f = n * PI2 - PI  # [-pi..pi]
        return torch.stack((torch.cos(f), torch.sin(f)), -1)


class AngleReparametrization:
    def __init__(self, interval: Tuple[float, float] = (-np.pi, np.pi)) -> None:
        r = torch.tensor(interval[1] - interval[0])
        is_pihalf = torch.allclose(r, torch.tensor(PI * 0.5), atol=1e-3)
        is_pi = torch.allclose(r, torch.tensor(PI), atol=1e-3)
        is_2pi = torch.allclose(r, torch.tensor(2 * PI), atol=1e-3)
        assert (
            is_pihalf or is_pi or is_2pi
        ), "Constraint intervals must be of length pi/2, pi or 2pi"
        if is_pihalf:
            # map -1..1/-1..1 -> 0..1/0..1
            self.scale = torch.tensor([0.5, 0.5])
            self.loc = torch.tensor([0.5, 0.5])
            self.rot = torch.tensor(np.cos(interval[0]) + 1j * np.sin(interval[0]))
        elif is_pi:
            # map -1..1/-1..1 -> -1..1/0..1
            self.scale = torch.tensor([1.0, 0.5])
            self.loc = torch.tensor([0, 0.5])
            self.rot = torch.tensor(np.cos(interval[0]) + 1j * np.sin(interval[0]))
        else:
            # map -1..1/-1..1 ->     -1..1/-1..1
            self.scale = torch.tensor([1.0, 1.0])
            self.loc = torch.tensor([0, 0.0])
            self.rot = torch.tensor(1 + 0j)

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        z = F.tanh(z) * self.scale + self.loc
        z = torch.view_as_real(torch.view_as_complex(z) * self.rot)
        return F.normalize(z, dim=-1)

    def inv(self, angle: torch.Tensor) -> torch.Tensor:
        z = torch.stack((torch.cos(angle), torch.sin(angle)), -1)
        z = torch.view_as_real(torch.view_as_complex(z) * torch.conj(self.rot))
        z = (z - self.loc) / self.scale
        return z
