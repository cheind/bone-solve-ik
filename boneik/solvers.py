from functools import partial

import torch
import torch.optim as optim

from . import kinematics


def vanilla_bone_loss(
    graph: kinematics.SkeletonGraph,
    anchors: torch.FloatTensor,
    weights: torch.FloatTensor,
):
    fkt = kinematics.fk(graph)
    loss = (
        torch.square(anchors - fkt[:, :3, 3]).sum(-1) * weights
    ).sum() / weights.sum()

    return loss


class IKSolver:
    def __init__(
        self,
        graph: kinematics.SkeletonGraph,
        normalize: bool = True,
    ) -> None:

        self.graph = graph
        self.normalize = normalize
        self.loss_fn = vanilla_bone_loss
        self._init_params()

    def _init_params(self) -> None:
        self.params = []
        for _, _, bone in self.graph.edges(data="bone"):
            ps = [p for p in bone.parameters() if p.requires_grad]
            self.params.extend(ps)

    def _closure(
        self, opt: optim.LBFGS, anchors: torch.FloatTensor, weights: torch.FloatTensor
    ):
        opt.zero_grad()
        loss = self.loss_fn(self.graph, anchors, weights)
        loss.backward()
        return loss

    def solve(
        self,
        anchors: torch.FloatTensor,
        weights: torch.FloatTensor,
        max_epochs: int = 100,
        min_abs_change: float = 1e-5,
        lr: float = 1e0,
    ) -> float:
        last_loss, loss = 1e10, 1e10
        opt = optim.LBFGS(
            self.params, history_size=100, lr=lr, line_search_fn="strong_wolfe"
        )
        closure = partial(self._closure, opt=opt, anchors=anchors, weights=weights)
        for e in range(max_epochs):
            opt.step(closure)
            loss = self.loss_fn(self.graph, anchors, weights).item()
            if (last_loss - loss) < min_abs_change:
                break
            last_loss = loss
            self._normalize()
        print(f"Completed after {e+1} epochs, loss {loss}")
        return loss

    def _normalize(self):
        if self.normalize:
            for _, _, bone in self.graph.edges(data="bone"):
                bone: kinematics.Bone
                bone.project_()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import transformations as T

    from .reparametrizations import PI

    torch.autograd.set_detect_anomaly(True)

    gen = kinematics.SkeletonGenerator()
    gen.bone(
        "root",
        "j1",
        t_uv=T.translation_matrix([0, 1.0, 0]),
        rz=kinematics.RotZ(interval=(-PI / 8, PI / 8)),
    ).bone(
        "j1",
        "j2",
        t_uv=T.translation_matrix([0, 1.0, 0]),
        rz=kinematics.RotZ(interval=(-PI / 4, -PI / 8)),
    )

    graph = gen.create_graph(relabel_order=["root", "j1", "j2"])
    solver = IKSolver(graph)

    anchors = torch.zeros(3, 3)
    weights = torch.zeros(3)

    anchors[1] = torch.Tensor([1.0, 1.0, 0])
    weights[1] = 1.0

    anchors[2] = torch.Tensor([2.0, 0.0, 0])
    weights[2] = 1.0
    solver.solve(anchors, weights, lr=1e-1)
    print(kinematics.fmt_skeleton(graph))

    # Plot anchors
    fig, ax = plt.subplots()
    for a, w in zip(anchors, weights):
        if w > 0:
            ax.scatter([a[0].item()], [a[1].item()], c="k", marker="+")

    with torch.no_grad():
        fkt = kinematics.fk(graph)
        for u, v in graph.graph["bfs_edges"]:
            tu = fkt[u, :2, 3].numpy()
            tv = fkt[v, :2, 3].numpy()
            ax.plot([tu[0], tv[0]], [tu[1], tv[1]], c="green")
            ax.scatter([tu[0], tv[0]], [tu[1], tv[1]], c="green")

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
    plt.show()
