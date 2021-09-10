from typing import Dict, Optional, Tuple, List, Set, Union, Type

import torch
import torch.nn

from . import dofs


RangeConstraint = Optional[Tuple[float, float]]
DofDict = Dict[str, RangeConstraint]
DofSet = Set[str]


class Bone(torch.nn.Module):
    """Describes the geometric relationship between two joints.

    The bone is conceptionally an edge between two vertices (base, tip) of the
    kinematic graph. It stores the how the tip frame is rotated/translated wrt.
    the base. The transformation consists of an initial transformation followed
    by a delta transformation. While the initial transformation remains fixed
    throughout the optimization, the delta transformation offers up to 6DOF.
    Each DOF can be unlocked for optimization and constrained to a specific
    interval. Currently, only rotations support angle constraints.

    There will be a bone instance for every connected pair of joints (vertices)
    in the kinematic graph (see kinematics.Body). We store the bone as attribute
    of the edge connecting these joints. This allows several bones to be
    represented starting from a common joint.
    """

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

    def sample(self, nsamples: int = 1) -> torch.FloatTensor:
        """Sample Nx6 delta values."""

        def _sample_once():
            return [
                self.rx.sample(),
                self.ry.sample(),
                self.rz.sample(),
                self.tx.sample(),
                self.ty.sample(),
                self.tz.sample(),
            ]

        return torch.as_tensor([_sample_once() for _ in range(nsamples)]).view(-1, 6)

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
