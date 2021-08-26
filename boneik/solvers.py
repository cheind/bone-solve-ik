from functools import partial

import torch
import torch.optim as optim

from . import kinematics


def vanilla_bone_loss(
    graph: kinematics.SkeletonGraph,
    anchor_dict: kinematics.VertexTensorDict,
):
    loss = 0.0
    fk_dict = kinematics.fk(graph)
    for vertex, loc in anchor_dict.items():
        lss = ((loc - fk_dict[vertex][:3, 3]) ** 2).sum()
        loss += lss
    return loss


class IKSolver:
    def __init__(
        self,
        graph: kinematics.SkeletonGraph,
        reproject: bool = True,
    ) -> None:

        self.graph = graph
        self.reproject = reproject
        self.loss_fn = vanilla_bone_loss
        self._init_params()

    def _init_params(self) -> None:
        self.params = []
        for _, _, bone in self.graph.edges(data="bone"):
            ps = [p for p in bone.parameters() if p.requires_grad]
            self.params.extend(ps)

    def _closure(self, opt: optim.LBFGS, anchor_dict: kinematics.VertexTensorDict):
        opt.zero_grad()
        loss = self.loss_fn(self.graph, anchor_dict)
        loss.backward()
        return loss

    def solve(
        self,
        anchor_dict: kinematics.VertexTensorDict,
        max_epochs: int = 10,
        min_abs_change: float = 1e-5,
        lr: float = 1e0,
    ) -> float:
        last_loss, loss = 1e10, 1e10
        opt = optim.LBFGS(
            self.params, history_size=100, lr=lr, line_search_fn="strong_wolfe"
        )
        closure = partial(self._closure, anchor_dict=anchor_dict, opt=opt)
        for e in range(max_epochs):
            opt.step(closure)
            loss = self.loss_fn(self.graph, anchor_dict).item()
            if (last_loss - loss) < min_abs_change:
                break
            last_loss = loss
            self._reproject()
        print(f"Completed after {e+1} epochs, loss {loss}")
        return loss

    def _reproject(self):
        if self.reproject:
            for _, _, bone in self.graph.edges(data="bone"):
                bone: kinematics.Bone
                bone.project_()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import transformations as T

    from .reparametrizations import PI

    gen = kinematics.SkeletonGenerator()
    gen.bone(
        0,
        1,
        t_uv=T.translation_matrix([0, 1.0, 0]),
        rotz=kinematics.RotZ(interval=(-PI / 8, PI / 8)),
    ).bone(
        1,
        "end",
        t_uv=T.translation_matrix([0, 1.0, 0]),
        rotz=kinematics.RotZ(interval=(-(2 * PI) / 4, -PI / 4)),
    )

    graph = gen.create_graph()
    solver = IKSolver(graph)

    anchor_dict = {
        1: torch.Tensor([1.0, 1.0, 0]),
        "end": torch.Tensor([2.0, 0.0, 0]),
    }
    solver.solve(anchor_dict=anchor_dict, lr=1.0)
    print(kinematics.fmt_skeleton(graph))

    # Plot anchors
    fig, ax = plt.subplots()
    for n, loc in anchor_dict.items():
        ax.scatter([loc[0].item()], [loc[1].item()], c="k", marker="+")

    with torch.no_grad():
        fk_dict = kinematics.fk(graph)
        for u, v in graph.graph["bfs_edges"]:
            tu = fk_dict[u][:2, 3].numpy()
            tv = fk_dict[v][:2, 3].numpy()
            ax.plot([tu[0], tv[0]], [tu[1], tv[1]], c="green")
            ax.scatter([tu[0], tv[0]], [tu[1], tv[1]], c="green")

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
    plt.show()
