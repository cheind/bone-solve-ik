from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .reparametrizations import ConstrainedAngleReparametrization


class RotDOF(torch.nn.Module):
    def __init__(
        self,
        *,
        angle: float = 0.0,
        interval: Tuple[float, float] = None,
        unlocked: bool = True,
    ):
        super().__init__()
        self.reparam = ConstrainedAngleReparametrization(interval)
        self.uangle = Parameter(self.reparam.angle2log(angle), requires_grad=unlocked)

    @property
    def angle(self):
        return self.reparam.log2angle(self.uangle)

    def unlock(self, interval: Tuple[float, float] = None):
        if interval is not None:
            angle = self.angle
            self.reparam = ConstrainedAngleReparametrization(interval)
            self.uangle.data[:] = self.reparam.angle2log(angle)
        self.uangle.requires_grad_(True)

    def matrix(self) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def project(self):
        self.uangle.data.clamp_(-1.0, 1.0)


class RotX(RotDOF):
    def matrix(self) -> torch.Tensor:
        c, s = self.reparam.exp(self.uangle)
        m = torch.eye(4)
        m[1, 1] = c
        m[1, 2] = -s
        m[2, 1] = s
        m[2, 2] = c
        return m


class RotY(RotDOF):
    def matrix(self) -> torch.Tensor:
        c, s = self.reparam.exp(self.uangle)
        m = torch.eye(4)
        m[0, 0] = c
        m[0, 2] = s
        m[2, 0] = -s
        m[2, 2] = c
        return m


class RotZ(RotDOF):
    def matrix(self) -> torch.Tensor:
        c, s = self.reparam.exp(self.uangle)
        m = torch.eye(4)
        m[0, 0] = c
        m[0, 1] = -s
        m[1, 0] = s
        m[1, 1] = c
        return m
