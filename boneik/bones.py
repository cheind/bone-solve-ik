from typing import Dict, Optional, Tuple, List, Set

import torch
import torch.nn

from . import dofs


RangeConstraint = Optional[Tuple[float, float]]
DofDict = Dict[str, RangeConstraint]
DofSet = Set[str]


class Bone(torch.nn.Module):
    def __init__(
        self, tip_to_base: torch.FloatTensor, dof_dict: Dict[str, RangeConstraint]
    ) -> None:
        super().__init__()
        self.tip_to_base = torch.as_tensor(tip_to_base).float()
        self._create_dofs(dof_dict or {})

    def _create_dofs(self, dof_dict: Dict[str, RangeConstraint]):
        the_dofs = []
        for name, klass in zip(dofs.DOF_NAMES, dofs.DOF_CLASSES):
            if name in dof_dict:
                the_dofs.append(
                    klass(value=0.0, interval=dof_dict[name], unlocked=True)
                )
            else:
                the_dofs.append(klass(value=0.0, interval=None, unlocked=False))
        self.dofs = torch.nn.ModuleList(the_dofs)

    def matrix(self) -> torch.FloatTensor:
        return self.delta_matrix() @ self.tip_to_base

    def delta_matrix(self) -> torch.FloatTensor:
        d = torch.eye(4)
        for dof in self.dofs:
            d = dof.matrix() @ d
        return d

    def set_delta(self, ds: List[float]):
        for d, dof in zip(ds, self.dofs):
            dof.set_value(d)

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
