import matplotlib.pyplot as plt
import transformations as T
import torch

from boneik import kinematics, solvers
from boneik.reparametrizations import PI


def draw(ax, graph):
    with torch.no_grad():
        fk_dict = kinematics.fk(graph)
        for u, v in graph.graph["bfs_edges"]:
            tu = fk_dict[u][:2, 3].numpy()
            tv = fk_dict[v][:2, 3].numpy()
            ax.plot([tu[0], tv[0]], [tu[1], tv[1]], c="green")
            ax.scatter([tv[0]], [tv[1]], c="green")
    ax.scatter([0], [0], c="k")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1, 5)


def main():
    gen = kinematics.SkeletonGenerator()
    gen.bone(
        0,
        1,
        t_uv=T.translation_matrix([0, 1.5, 0]),
        rotz=kinematics.RotZ(interval=(-PI / 2, PI / 2)),
    )
    gen.bone(
        1,
        2,
        t_uv=T.translation_matrix([0, 1.0, 0]),
        rotz=kinematics.RotZ(interval=(-PI / 2, PI / 2)),
    )
    gen.bone(
        2,
        3,
        t_uv=T.translation_matrix([0, 0.5, 0]),
        rotz=kinematics.RotZ(interval=(-PI / 2, PI / 2)),
    )
    gen.bone(
        3,
        4,
        t_uv=T.translation_matrix([0, 0.5, 0]),
        rotz=kinematics.RotZ(interval=(-PI / 2, PI / 2)),
    )
    graph = gen.create_graph()
    N = graph.number_of_nodes()
    solver = solvers.IKSolver(graph)
    fig, ax = plt.subplots()

    anchors = torch.zeros(N, 3)
    weights = torch.zeros(N)

    def on_move(event):
        if not event.inaxes:
            return
        loc = torch.tensor([event.xdata, event.ydata, 0]).float()
        anchors[-1] = loc
        weights[-1] = 1.0
        solver.solve(anchors, weights, lr=1e-1)
        ax.cla()
        draw(event.inaxes, graph)
        fig.canvas.draw_idle()

    draw(ax, graph)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.show()


if __name__ == "__main__":
    main()