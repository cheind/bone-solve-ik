from typing import Optional, List, Tuple, Dict
import numpy as np
import torch
import torch.nn
import transformations as T
import torch.optim as optim
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


class Bone(torch.nn.Module):
    def __init__(
        self,
        name: str,
        children: Optional[List["Bone"]] = None,
        t_rest: torch.Tensor = torch.eye(4),
    ):
        super().__init__()
        self.name = name
        self.child_bones = torch.nn.ModuleList(children)
        self.t_rest = t_rest
        self.rot_x = RotX()
        self.rot_y = RotY()
        self.rot_z = RotZ()

    def matrix(self):
        return (
            self.t_rest
            @ self.rot_x.matrix()
            @ self.rot_y.matrix()
            @ self.rot_z.matrix()
        )

    def link_to(self, other: "Bone"):
        self.child_bones.append(other)
        return other

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other: "Bone"):
        return self.name == other.name


def bfs(root: Bone):
    """Breadth-first search over kinematic starting at root."""
    queue = [(root, None)]
    while queue:
        b, parent = queue.pop(0)
        yield b, parent
        queue.extend([(c, b) for c in b.child_bones])


def fk(root: Bone, root_pose: torch.Tensor = None) -> Dict[Bone, torch.Tensor]:
    """Computes the forward kinematic poses of each bone."""
    if root_pose is None:
        root_pose = torch.eye(4)
    fk_dict = {}
    for bone, parent in bfs(root):
        if parent is None:
            fk_dict[bone] = root_pose @ bone.matrix()
        else:
            fk_dict[bone] = fk_dict[parent] @ bone.matrix()
    return fk_dict


def vanilla_bone_loss(root: Bone, anchor_dict: Dict[Bone, torch.Tensor]):
    loss = 0.0
    fk_dict = fk(root)
    for bone, loc in anchor_dict.items():
        loss += ((loc - fk_dict[bone][:3, 3]) ** 2).sum()
    return loss


def solve(
    root: Bone,
    anchor_dict: Dict[Bone, torch.Tensor],
    max_epochs: int = 20,
    min_rel_change: float = 1e-2,
):
    opt = optim.LBFGS(
        [p for p in root.parameters() if p.requires_grad], history_size=10, max_iter=4
    )
    last_loss = 1e10
    for e in range(max_epochs):

        def closure():
            opt.zero_grad()
            loss = vanilla_bone_loss(root, anchor_dict)
            loss.backward()
            return loss

        opt.step(closure)
        loss = vanilla_bone_loss(root, anchor_dict).item()
        if loss >= last_loss or (last_loss - loss) / last_loss < min_rel_change:
            break
        last_loss = loss
    print(f"Completed after {e} epochs, loss {loss}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    b0 = Bone("0")
    b1 = Bone("1", t_rest=torch.Tensor(T.translation_matrix([0, 1.0, 0])))
    b2 = Bone("end", t_rest=torch.Tensor(T.translation_matrix([0, 1.0, 0])))

    b0.rot_z.unlock((-0.5, 0.5))
    b1.rot_z.unlock()
    b0.link_to(b1).link_to(b2)

    a_dict = {b1: torch.Tensor([1.0, 1.0, 0]), b2: torch.Tensor([2.0, 0.0, 0])}
    solve(b0, a_dict)

    # Plot anchors
    fig, ax = plt.subplots()
    for n, loc in a_dict.items():
        ax.scatter([loc[0].item()], [loc[1].item()], c="k", marker="+")

    with torch.no_grad():
        fk_dict = fk(b0)
        for bone, parent in bfs(b0):
            print(bone.name, bone.rot_z.angle)
            tb = fk_dict[bone][:2, 3].numpy()
            if parent is not None:
                tp = fk_dict[parent][:2, 3].numpy()
                ax.plot([tp[0], tb[0]], [tp[1], tb[1]], c="green")
            ax.scatter([tb[0]], [tb[1]], c="green")

        ax.set_xlim(-5, 5)
        ax.set_ylim(-1, 5)
        plt.show()
