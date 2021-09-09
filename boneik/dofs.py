from typing import Tuple

import torch
from torch.nn.parameter import Parameter

from .reparametrizations import ConstrainedAngleReparametrization


class BaseRotDOF(torch.nn.Module):
    def __init__(
        self,
        *,
        value: float = 0.0,
        interval: Tuple[float, float] = None,
        unlocked: bool = True,
    ):
        super().__init__()
        self.reparam = ConstrainedAngleReparametrization(interval)
        self.uangle = Parameter(
            self.reparam.angle2log(value).float(), requires_grad=unlocked
        )
        self._reset_value = self.uangle.data.clone()

    @property
    def value(self):
        return self.reparam.log2angle(self.uangle)

    def matrix(self) -> torch.Tensor:
        raise NotImplementedError

    def project_(self):
        self.reparam.project_inplace(self.uangle)

    def reset_(self):
        self.uangle.data[:] = self._reset_value

    def set_value(self, angle: float):
        self.uangle.data[:] = self.reparam.angle2log(angle).float()


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


class BaseTransDOF(torch.nn.Module):
    def __init__(
        self,
        *,
        value: float = 0.0,
        interval: Tuple[float, float] = None,
        unlocked: bool = True,
    ):
        super().__init__()
        self.offset = Parameter(torch.tensor(value).float(), requires_grad=unlocked)
        self._reset_value = self.offset.data.clone()

    def matrix(self) -> torch.Tensor:
        raise NotImplementedError

    def reset_(self):
        self.offset.data.fill_(self._reset_value)

    def project_(self):
        pass

    @property
    def value(self) -> torch.FloatTensor:
        return self.offset

    def set_value(self, offset: float):
        self.offset.data.fill_(offset)


class TransX(BaseTransDOF):
    def matrix(self) -> torch.Tensor:
        m = torch.eye(4)
        m[0, 3] = self.offset
        return m


class TransY(BaseTransDOF):
    def matrix(self) -> torch.Tensor:
        m = torch.eye(4)
        m[1, 3] = self.offset
        return m


class TransZ(BaseTransDOF):
    def matrix(self) -> torch.Tensor:
        m = torch.eye(4)
        m[2, 3] = self.offset
        return m


DOF_NAMES = ["rx", "ry", "rz", "tx", "ty", "tz"]
DOF_CLASSES = [RotX, RotY, RotZ, TransX, TransY, TransZ]