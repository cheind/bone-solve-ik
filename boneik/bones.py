from typing import Dict, Optional

import torch
import torch.nn

from .dofs import RotX, RotY, RotZ


class BoneDOF(torch.nn.Module):
    def __init__(
        self,
        rotx: Optional[RotX] = None,
        roty: Optional[RotY] = None,
        rotz: Optional[RotZ] = None,
    ):
        super().__init__()
        if rotx is None:
            rotx = RotX(unlocked=False)
        if roty is None:
            roty = RotX(unlocked=False)
        if rotz is None:
            rotz = RotZ(unlocked=False)
        self.rotx = rotx
        self.roty = roty
        self.rotz = rotz

    def matrix(self):
        return self.rotx.matrix() @ self.roty.matrix() @ self.rotz.matrix()

    def project(self):
        self.rotx.project()
        self.roty.project()
        self.rotz.project()


class Bone:
    def __init__(
        self,
        name: str,
        t: torch.Tensor = None,
    ):
        super().__init__()
        self.name = name
        self.children = []
        self.parent = None
        if t is None:
            t = torch.eye(4)
        self.t = t

    def matrix(self):
        return self.t

    def link_to(self, other: "Bone"):
        self.children.append(other)
        other.parent = self
        return other

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other: "Bone"):
        return self.name == other.name

    def __str__(self) -> str:
        return f"Bone(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()


BoneTensorDict = Dict[Bone, torch.Tensor]
BoneDOFDict = Dict[Bone, BoneDOF]


def bfs(root: Bone):
    """Breadth-first search over kinematic starting at root."""
    queue = [root]
    while queue:
        b = queue.pop(0)
        yield b
        queue.extend([c for c in b.children])


def fk(root: Bone, dof_dict: BoneDOFDict = None) -> BoneTensorDict:
    """Computes the forward kinematic poses of each bone."""
    fk_dict = {}
    for bone in bfs(root):
        fk_prev = torch.eye(4) if bone.parent is None else fk_dict[bone.parent]
        t_dof = torch.eye(4)
        if bone in dof_dict:
            t_dof = dof_dict[bone].matrix()
        fk_dict[bone] = fk_prev @ bone.matrix() @ t_dof
    return fk_dict
