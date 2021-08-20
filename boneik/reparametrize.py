from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

PI = np.pi
PI2 = 2 * PI


class ConstrainedAngleReparametrization:
    """Reparametrize an contrained angle for unconstrained optimization."""

    def __init__(self, interval: Optional[Tuple[float, float]] = None) -> None:
        if interval is None:
            self.i = (-PI, PI)
            self.length = 2 * PI
            self.is_constrained = False
        else:
            self.i = interval
            self.length = self.i[1] - self.i[0]
            self.is_constrained = True
        assert (
            interval is None or self.length <= PI
        ), "Constrained interval must be <= PI."
        self._init_affine()

    @torch.no_grad()
    def _init_affine(self):
        if self.is_constrained:
            theta = self.length / 2
            scale = torch.eye(3)
            scale[0, 0] = (1 - np.cos(theta)) / 2
            scale[1, 1] = (self.length) / 2

            trans = torch.eye(3)
            trans[0, 2] = -(-1) * scale[0, 0] + np.cos(theta)
            trans[1, 2] = -(-1) * scale[1, 1] + -np.sin(theta)

            rot = torch.eye(3)
            theta_diff = self.i[0] - (-theta)
            rot[0, 0] = np.cos(theta_diff)
            rot[1, 1] = np.cos(theta_diff)
            rot[0, 1] = -np.sin(theta_diff)
            rot[1, 0] = np.sin(theta_diff)
            self.affine = rot @ trans @ scale
        else:
            self.affine = torch.eye(3)
        self.inv_affine = torch.inverse(self.affine)

    def exp(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(z)
        zh = torch.ones(3, 1)
        zh[:2, 0] = z
        z = (self.affine @ zh)[:2, 0]
        z = F.normalize(z, dim=-1)
        return z

    def log(self, z: torch.Tensor) -> torch.Tensor:
        zh = torch.ones(3, 1)
        zh[:2, 0] = z
        z = (self.inv_affine @ zh)[:2, 0]
        return z

    @torch.no_grad()
    def angle2log(self, theta: torch.Tensor) -> torch.Tensor:
        theta = torch.as_tensor(theta)
        z = torch.tensor([torch.cos(theta), torch.sin(theta)])
        return self.log(z)

    @torch.no_grad()
    def log2angle(self, z: torch.Tensor) -> torch.Tensor:
        e = self.exp(z)
        return torch.atan2(e[1], e[0])
