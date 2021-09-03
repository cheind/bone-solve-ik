import torch
from . import kinematics


def draw_axis(ax3d, t_world: torch.Tensor, length: float = 0.5, lw: float = 1.0):
    p_f = torch.tensor(
        [[0, 0, 0, 1], [length, 0, 0, 1], [0, length, 0, 1], [0, 0, length, 1]]
    ).T
    p_0 = t_world @ p_f
    X = torch.stack([p_0[:, 0].T, p_0[:, 1].T], 0).numpy()
    Y = torch.stack([p_0[:, 0].T, p_0[:, 2].T], 0).numpy()
    Z = torch.stack([p_0[:, 0].T, p_0[:, 3].T], 0).numpy()
    ax3d.plot3D(X[:, 0], X[:, 1], X[:, 2], "r-", linewidth=lw)
    ax3d.plot3D(Y[:, 0], Y[:, 1], Y[:, 2], "g-", linewidth=lw)
    ax3d.plot3D(Z[:, 0], Z[:, 1], Z[:, 2], "b-", linewidth=lw)


@torch.no_grad()
def draw(
    ax3d,
    graph: kinematics.SkeletonGraph,
    anchors: torch.Tensor = None,
    draw_local_frames: bool = True,
    draw_vertex_labels: bool = False,
    hide_root: bool = True,
):
    fkt = kinematics.fk(graph)
    root = graph.graph["root"]
    for u, v in graph.graph["bfs_edges"]:
        if u == root and hide_root:
            continue
        x = fkt[[u, v], 0, 3].numpy()
        y = fkt[[u, v], 1, 3].numpy()
        z = fkt[[u, v], 2, 3].numpy()
        ax3d.plot(x, y, z, lw=1, c="k", linestyle="--")
        ax3d.scatter(x[1], y[1], z[1], c="k")
    if draw_local_frames:
        if not hide_root:
            draw_axis(ax3d, fkt[root])
        for _, v in graph.graph["bfs_edges"]:
            draw_axis(ax3d, fkt[v])
    if draw_vertex_labels:
        for u, label in graph.nodes.data("label"):
            if u == root and hide_root:
                continue
            xyz = fkt[u, :3, 3].numpy()
            ax3d.text(xyz[0], xyz[1], xyz[2], label, fontsize="x-small")

    if anchors is not None:
        anchors = anchors.numpy()
        ax3d.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], c="r", marker="o")
        # for u, label in graph.nodes.data("label"):
        #     ax3d.text(
        #         anchors[u, 0], anchors[u, 1], anchors[u, 2], label, fontsize="x-small"
        #     )
