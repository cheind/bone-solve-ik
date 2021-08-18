from typing import Tuple

import numpy as np
import torch
from torch.nn.parameter import Parameter

from .reparametrize import AngleReparametrization


class RotDOF(torch.nn.Module):
    def __init__(
        self,
        angle: float = 0.0,
        constraint_interval: Tuple[float, float] = (-np.pi, np.pi),
        unlocked: bool = False,
    ):
        super().__init__()
        self.reparam = AngleReparametrization(constraint_interval)
        self.uangle = Parameter(
            self.reparam.inv(torch.tensor([angle])), requires_grad=unlocked
        )

    @property
    def angle(self):
        return self.reparam(self.uangle)

    def unlock(self, constraint_interval: Tuple[float, float] = None):
        if constraint_interval is not None:
            angle = self.reparam(self.uangle)
            self.reparam = AngleReparametrization(constraint_interval)
            self.uangle.data[:] = self.reparam.inv(torch.tensor([angle]))
        self.uangle.requires_grad_(True)

    def matrix(self) -> torch.Tensor:
        raise NotImplementedError


class RotX(RotDOF):
    def matrix(self) -> torch.Tensor:
        a = self.angle
        c = torch.cos(a)
        s = torch.sin(a)
        m = torch.eye(4)
        m[1, 1] = c
        m[1, 2] = -s
        m[2, 1] = s
        m[2, 2] = c
        return m


class RotY(RotDOF):
    def matrix(self) -> torch.Tensor:
        a = self.angle
        c = torch.cos(a)
        s = torch.sin(a)
        m = torch.eye(4)
        m[0, 0] = c
        m[0, 2] = s
        m[2, 0] = -s
        m[2, 2] = c
        return m


class RotZ(RotDOF):
    def matrix(self) -> torch.Tensor:
        a = self.angle
        c = torch.cos(a)
        s = torch.sin(a)
        m = torch.eye(4)
        m[0, 0] = c
        m[0, 1] = -s
        m[1, 0] = s
        m[1, 1] = c
        return m
