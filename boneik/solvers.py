from functools import partial

import torch
import torch.optim as optim
import logging

from . import kinematics
from . import criteria

_logger = logging.getLogger("boneik")


def _closure_step(
    kinematic: kinematics.KinematicGraph, opt: optim.LBFGS, crit: criteria.Criterium
):
    opt.zero_grad()
    loss = crit(kinematic)
    loss.backward()
    return loss


class IKSolver:
    def __init__(self, kinematic: kinematics.KinematicGraph) -> None:

        self.kinematic = kinematic
        self._init_params()

    def _init_params(self) -> None:
        self.params = []
        for _, _, bone in self.kinematic.graph.edges(data="bone"):
            ps = [p for p in bone.parameters() if p.requires_grad]
            self.params.extend(ps)

    def solve(
        self,
        crit: criteria.Criterium,
        max_epochs: int = 100,
        min_abs_change: float = 1e-5,
        lr: float = 1e0,
        history_size: int = 100,
        max_iter: int = 20,
        reproject: bool = True,
    ) -> float:
        last_loss, loss = 1e10, 1e10
        opt = optim.LBFGS(
            self.params,
            history_size=history_size,
            max_iter=max_iter,
            lr=lr,
            line_search_fn="strong_wolfe",
        )
        closure = partial(_closure_step, opt=opt, crit=crit, kinematic=self.kinematic)
        for e in range(max_epochs):
            opt.step(closure)
            loss = crit(self.kinematic).item()
            if (last_loss - loss) < min_abs_change:
                break
            last_loss = loss
            if reproject:
                self._reproject()
        _logger.debug(f"Completed after {e+1} epochs, loss {loss}")
        return loss

    def _reproject(self):
        for _, _, bone in self.kinematic.graph.edges(data="bone"):
            bone: kinematics.Bone
            bone.project_()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import transformations as T

    from .reparametrizations import PI
    from . import criteria
    from . import draw

    b = kinematics.KinematicBuilder()

    b.add_bone(
        "root",
        "j1",
        tip_to_base=T.translation_matrix([0, 1.0, 0]),
        dofs={"rz": (-PI / 4, PI / 4)},
    ).add_bone(
        "j1",
        "j2",
        tip_to_base=T.translation_matrix([0, 1.0, 0]),
        dofs={"rz": (-PI / 4, -PI / 8)},
    )
    k = b.finalize(["root", "j1", "j2"])

    solver = IKSolver(k)

    anchors = torch.zeros(3, 3)
    weights = torch.zeros(3)

    anchors[1] = torch.Tensor([1.0, 1.0, 0])
    weights[1] = 0.1

    anchors[2] = torch.Tensor([2.0, 0.0, 0])
    weights[2] = 1.0
    print(k)
    loss = solver.solve(criteria.EuclideanDistanceCriterium(anchors, weights), lr=1e-1)
    print(k)
    print("loss", loss)

    fig, ax = draw.create_figure3d(axes_ranges=[[-2, 2], [-2, 2], [0, 1]])
    draw.draw_kinematics(ax, kinematic=k, anchors=anchors, draw_root=True)
    plt.show()
