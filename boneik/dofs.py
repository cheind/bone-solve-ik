from typing import Tuple

import torch
from torch.nn.parameter import Parameter

from .reparametrizations import ConstrainedAngleReparametrization


class BaseRotDOF(torch.nn.Module):
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

    def matrix(self) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def project_(self):
        self.reparam.project_inplace(self.uangle)


class RotX(BaseRotDOF):
    def matrix(self) -> torch.Tensor:
        c, s = self.reparam.exp(self.uangle)
        m = torch.eye(4)
        m[1, 1] = c
        m[1, 2] = -s
        m[2, 1] = s
        m[2, 2] = c
        return m


class RotY(BaseRotDOF):
    def matrix(self) -> torch.Tensor:
        c, s = self.reparam.exp(self.uangle)
        m = torch.eye(4)
        m[0, 0] = c
        m[0, 2] = s
        m[2, 0] = -s
        m[2, 2] = c
        return m


class RotZ(BaseRotDOF):
    def matrix(self) -> torch.Tensor:
        c, s = self.reparam.exp(self.uangle)
        m = torch.eye(4)
        m[0, 0] = c
        m[0, 1] = -s
        m[1, 0] = s
        m[1, 1] = c
        return m
