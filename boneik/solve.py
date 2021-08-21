from functools import partial
import torch
import torch.optim as optim
from . import bones


def vanilla_bone_loss(
    root: bones.Bone, dof_dict: bones.BoneDOFDict, anchor_dict: bones.BoneTensorDict
):
    loss = 0.0
    fk_dict = bones.fk(root, dof_dict=dof_dict)
    for bone, loc in anchor_dict.items():
        lss = ((loc - fk_dict[bone][:3, 3]) ** 2).sum()
        loss += lss
    return loss


class IKSolver:
    def __init__(
        self,
        root: bones.Bone,
        dof_dict: bones.BoneDOFDict,
        lr: float = 1e0,
    ) -> None:

        self.root = root
        self.dof_dict = dof_dict
        self.loss_fn = vanilla_bone_loss
        self._init_optimizer(dof_dict, lr)

    def _init_optimizer(self, dof_dict: bones.BoneDOFDict, lr: float) -> None:
        params = []
        for dof in dof_dict.values():
            params.extend([p for p in dof.parameters() if p.requires_grad])
        self.opt = optim.LBFGS(params, history_size=10, max_iter=4, lr=lr)

    def _closure(self, anchor_dict: bones.BoneTensorDict):
        self.opt.zero_grad()
        loss = self.loss_fn(self.root, self.dof_dict, anchor_dict)
        loss.backward()
        return loss

    def solve(
        self,
        anchor_dict: bones.BoneTensorDict,
        max_epochs: int = 100,
        min_rel_change: float = 1e-4,
    ) -> float:
        last_loss, loss = 1e10, 1e10
        closure = partial(self._closure, anchor_dict=anchor_dict)
        for e in range(max_epochs):
            self.opt.step(closure)
            loss = self.loss_fn(self.root, self.dof_dict, anchor_dict).item()
            if loss >= last_loss or (last_loss - loss) / last_loss < min_rel_change:
                break
            last_loss = loss
        print(f"Completed after {e+1} epochs, loss {loss}")
        return loss


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import transformations as T
    from .reparametrize import PI

    b0 = bones.Bone("0")
    b1 = bones.Bone("1", t=torch.Tensor(T.translation_matrix([0, 1.0, 0])))
    b2 = bones.Bone("end", t=torch.Tensor(T.translation_matrix([0, 1.0, 0])))
    b0.link_to(b1).link_to(b2)

    dof_dict = {
        b0: bones.BoneDOF(rotz=bones.RotZ(interval=(-PI / 4, PI / 4))),  #
        b1: bones.BoneDOF(rotz=bones.RotZ()),
    }
    solver = IKSolver(b0, dof_dict=dof_dict)

    anchor_dict = {
        b1: torch.Tensor([1.2, 1.0, 0]),
        b2: torch.Tensor([2.0, 0.0, 0]),
    }

    anchor_dict = {
        # b1: torch.Tensor([1.0, 1.0, 0]),
        # b2: torch.Tensor([2.0, 0.0, 0]),
        b2: torch.tensor([-3.6342, -3.9752, 0.0000])
    }
    solver.solve(anchor_dict=anchor_dict)

    # Plot anchors
    fig, ax = plt.subplots()
    for n, loc in anchor_dict.items():
        ax.scatter([loc[0].item()], [loc[1].item()], c="k", marker="+")

    with torch.no_grad():
        fk_dict = bones.fk(b0, dof_dict)
        for bone in bones.bfs(b0):
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
