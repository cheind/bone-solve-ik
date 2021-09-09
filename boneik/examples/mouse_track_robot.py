import matplotlib.pyplot as plt
import transformations as T
import torch

from boneik import kinematics, solvers, criteria
from boneik.reparametrizations import PI


def draw(ax, k: kinematics.Kinematic):
    with torch.no_grad():
        fk = k.fk()
        for u, v in k.bfs_edges:
            tu = fk[u][:2, 3].numpy()
            tv = fk[v][:2, 3].numpy()
            ax.plot([tu[0], tv[0]], [tu[1], tv[1]], c="green")
            ax.scatter([tv[0]], [tv[1]], c="green")
    ax.scatter([0], [0], c="k")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1, 5)


def main():
    b = kinematics.KinematicBuilder()
    b.add_bone(
        0,
        1,
        tip_to_base=T.translation_matrix([0, 1.5, 0]),
        dofs={"rz": (-PI / 2, PI / 2)},
    ).add_bone(
        1,
        2,
        tip_to_base=T.translation_matrix([0, 1.0, 0]),
        dofs={"rz": (-PI / 2, PI / 2)},
    ).add_bone(
        2,
        3,
        tip_to_base=T.translation_matrix([0, 0.5, 0]),
        dofs={"rz": (-PI / 2, PI / 2)},
    ).add_bone(
        3,
        4,
        tip_to_base=T.translation_matrix([0, 0.5, 0]),
        dofs={"rz": (-PI / 2, PI / 2)},
    )
    k = b.finalize()
    N = k.graph.number_of_nodes()
    solver = solvers.IKSolver(k)
    fig, ax = plt.subplots()

    anchors = torch.zeros(N, 3)
    weights = torch.zeros(N)
    crit = criteria.EuclideanDistanceCriterium(anchors, weights)

    def on_move(event):
        if not event.inaxes:
            return
        loc = torch.tensor([event.xdata, event.ydata, 0]).float()
        crit.anchors[-1] = loc
        crit.weights[-1] = 1.0
        loss = solver.solve(
            crit, min_abs_change=0.01, history_size=5, max_iter=5, lr=1e-1
        )
        if loss > 1.0:
            k.reset_()
            solver.solve(crit, lr=1e-1)
        ax.cla()
        draw(event.inaxes, k)
        fig.canvas.draw_idle()

    draw(ax, k)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.show()


if __name__ == "__main__":
    main()