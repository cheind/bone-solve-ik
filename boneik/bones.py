from typing import Dict, Optional, Tuple, List, Set, Union, Type

import torch
import torch.nn

from . import dofs


RangeConstraint = Optional[Tuple[float, float]]
DofDict = Dict[str, RangeConstraint]
DofSet = Set[str]


class Bone(torch.nn.Module):
    def __init__(self, tip_to_base: torch.FloatTensor, dof_dict: DofDict) -> None:
        super().__init__()
        self.tip_to_base = torch.as_tensor(tip_to_base).float()
        self._create_dofs(dof_dict or {})

    def _create_dofs(self, dof_dict: DofDict):
        def _create(klass: Type, name: str, dof_dict):
            if name in dof_dict:
                the_dof = klass(value=0.0, interval=dof_dict[name], unlocked=True)
            else:
                the_dof = klass(value=0.0, interval=None, unlocked=False)
            return the_dof

        self.rx = _create(dofs.RotX, "rx", dof_dict)
        self.ry = _create(dofs.RotY, "ry", dof_dict)
        self.rz = _create(dofs.RotZ, "rz", dof_dict)
        self.tx = _create(dofs.TransX, "tx", dof_dict)
        self.ty = _create(dofs.TransY, "ty", dof_dict)
        self.tz = _create(dofs.TransZ, "tz", dof_dict)
        self.dofs = [self.rx, self.ry, self.rz, self.tx, self.ty, self.tz]

    def matrix(self) -> torch.FloatTensor:
        return self.delta_matrix() @ self.tip_to_base

    def delta_matrix(self) -> torch.FloatTensor:
        return (
            self.tx.matrix()
            @ self.ty.matrix()
            @ self.tz.matrix()
            @ self.rx.matrix()
            @ self.ry.matrix()
            @ self.rz.matrix()
        )

    def set_delta(self, ds: Union[List[float], torch.FloatTensor]):
        self.rx.set_value(ds[0])
        self.ry.set_value(ds[1])
        self.rz.set_value(ds[2])
        self.tx.set_value(ds[3])
        self.ty.set_value(ds[4])
        self.tz.set_value(ds[5])

    def get_delta(self):
        return [dof.value for dof in self.dofs]

    def project_(self):
        for dof in self.dofs:
            dof.project_()

    def reset_(self):
        for dof in self.dofs:
            dof.reset_()

    def __str__(self):
        return (
            "Bone(tip_to_base=\n"
            f"{self.tip_to_base}"
            ",\n"
            f"delta={[d.item() for d in self.get_delta()]}"
            ")"
        )
