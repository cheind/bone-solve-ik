from typing import Optional, List, Tuple, Dict
import numpy as np
import torch
import torch.nn
import transformations as T
import torch.optim as optim

from .dof import RotX, RotY, RotZ
from .reparametrize import PI


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

    def normalize(self):
        self.rotx.normalize()
        self.roty.normalize()
        self.rotz.normalize()


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


def vanilla_bone_loss(root: Bone, dof_dict: BoneDOFDict, anchor_dict: BoneTensorDict):
    loss = 0.0
    fk_dict = fk(root, dof_dict=dof_dict)
    for bone, loc in anchor_dict.items():
        l = ((loc - fk_dict[bone][:3, 3]) ** 2).sum()
        if bone in dof_dict:
            print(bone.name, l.item(), dof_dict[bone].rotz.matrix())
        loss += l
    return loss


def solve(
    root: Bone,
    dof_dict: BoneDOFDict,
    anchor_dict: BoneTensorDict,
    max_epochs: int = 100,
    min_rel_change: float = 1e-5,
    lr: float = 1e-2,
):
    params = []
    for dof in dof_dict.values():
        params.extend([p for p in dof.parameters() if p.requires_grad])

    opt = optim.LBFGS(params, history_size=10, max_iter=4, lr=lr)
    last_loss = 1e10
    for e in range(max_epochs):

        def closure():
            opt.zero_grad()
            loss = vanilla_bone_loss(root, dof_dict, anchor_dict)
            loss.backward()
            return loss

        opt.step(closure)
        # normalize
        loss = vanilla_bone_loss(root, dof_dict, anchor_dict).item()
        print(loss)
        if loss >= last_loss or (last_loss - loss) / last_loss < min_rel_change:
            break
        last_loss = loss
    print(f"Completed after {e+1} epochs, loss {loss}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    b0 = Bone("0")
    b1 = Bone("1", t=torch.Tensor(T.translation_matrix([0, 1.0, 0])))
    b2 = Bone("end", t=torch.Tensor(T.translation_matrix([0, 1.0, 0])))
    b0.link_to(b1).link_to(b2)

    dof_dict = {
        b0: BoneDOF(rotz=RotZ(interval=(-PI / 4, PI / 4))),  #
        b1: BoneDOF(rotz=RotZ()),
    }

    anchor_dict = {
        b1: torch.Tensor([1.2, 1.0, 0]),
        b2: torch.Tensor([2.0, 0.0, 0]),
    }

    solve(b0, dof_dict, anchor_dict)

    # anchor_dict = {
    #     # b1: torch.Tensor([1.0, 1.0, 0]),
    #     # b2: torch.Tensor([2.0, 0.0, 0]),
    #     b2: torch.tensor([-3.6342, -3.9752, 0.0000])
    # }
    # solve(b0, dof_dict, anchor_dict)

    # Plot anchors
    fig, ax = plt.subplots()
    for n, loc in anchor_dict.items():
        ax.scatter([loc[0].item()], [loc[1].item()], c="k", marker="+")

    with torch.no_grad():
        fk_dict = fk(b0, dof_dict)
        for bone in bfs(b0):
            if bone in dof_dict:
                print(
                    bone.name,
                    dof_dict[bone].rotz.angle,
                    dof_dict[bone].rotz.uangle,
                    dof_dict[bone].rotz.uangle.grad,
                )
            tb = fk_dict[bone][:2, 3].numpy()
            if bone.parent is not None:
                tp = fk_dict[bone.parent][:2, 3].numpy()
                ax.plot([tp[0], tb[0]], [tp[1], tb[1]], c="green")
            ax.scatter([tb[0]], [tb[1]], c="green")

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
    plt.show()
