from typing import Optional, List, Tuple, Dict
import numpy as np
import torch
import torch.nn
import transformations as T
import torch.optim as optim
from torch.nn.parameter import Parameter
import torch.distributions as dist
import torch.distributions.constraints as constraints


class RotDOF(torch.nn.Module):
    def __init__(
        self,
        angle: float = 0.0,
        angle_constraint: Tuple[float, float] = (-np.pi, np.pi),
        enabled: bool = False,
    ):
        super().__init__()
        self.constr = dist.transform_to(
            constraints.half_open_interval(angle_constraint[0], angle_constraint[1])
        )
        self.uangle = Parameter(
            self.constr.inv(torch.tensor([angle])), requires_grad=enabled
        )

    @property
    def angle(self):
        return self.constr(self.uangle)

    def enable(self, enable: bool):
        self.uangle.requires_grad_(enable)

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


class Bone(torch.nn.Module):
    def __init__(
        self,
        name: str,
        children: Optional[List["Bone"]] = None,
        t_parent: torch.Tensor = torch.eye(4),
    ):
        super().__init__()
        self.name = name
        self.child_bones = torch.nn.ModuleList(children)
        self.t_parent = t_parent
        self.rot_x = RotX()
        self.rot_y = RotY()
        self.rot_z = RotZ()

    def matrix(self):
        return (
            self.t_parent
            @ self.rot_x.matrix()
            @ self.rot_y.matrix()
            @ self.rot_z.matrix()
        )


def iterate_bones(root: Bone):
    queue = [(root, torch.eye(4))]
    while queue:
        b, t_world = queue.pop(0)
        t = t_world @ b.matrix()
        yield b, t
        queue.extend([(b, t) for b in b.child_bones])


def bone_loss(root: Bone, anchor_dict: Dict[str, torch.Tensor]):
    loss = 0.0
    for b, t in iterate_bones(root):
        if b.name in anchor_dict:
            loss += ((anchor_dict[b.name] - t[:3, 3]) ** 2).sum()
    return loss


def solve(root: Bone, anchor_dict: Dict[str, torch.Tensor]):

    opt = optim.Adam(
        [p for p in root.parameters() if p.requires_grad],
        lr=1e-1,
    )
    for _ in range(100):
        opt.zero_grad()
        loss = bone_loss(root, anchor_dict)
        loss.backward()
        opt.step()
        print(loss.item())


if __name__ == "__main__":
    b0 = Bone("0")
    b1 = Bone("1", t_parent=torch.Tensor(T.translation_matrix([0, 1.0, 0])))
    b2 = Bone("end", t_parent=torch.Tensor(T.translation_matrix([0, 1.0, 0])))

    b0.rot_z.enable(True)
    b1.rot_z.enable(True)

    b0.child_bones.append(b1)
    b1.child_bones.append(b2)

    solve(b0, {"end": torch.Tensor([2.0, 0.0, 0])})

    for b, t in iterate_bones(b0):
        print(b.name, b.rot_z.angle, t[:3, 3])
