from boneik.dofs import RotX
from boneik.reparametrizations import PI
import matplotlib.pyplot as plt
import transformations as T
import numpy as np
import torch
import pickle

from boneik import kinematics, solvers, utils, draw
from boneik import bvh


# def draw_axis(ax3d, t_world: torch.Tensor, length: float = 0.5, lw: float = 1.0):
#     p_f = torch.tensor(
#         [[0, 0, 0, 1], [length, 0, 0, 1], [0, length, 0, 1], [0, 0, length, 1]]
#     ).T
#     p_0 = t_world @ p_f
#     X = torch.stack([p_0[:, 0].T, p_0[:, 1].T], 0).numpy()
#     Y = torch.stack([p_0[:, 0].T, p_0[:, 2].T], 0).numpy()
#     Z = torch.stack([p_0[:, 0].T, p_0[:, 3].T], 0).numpy()
#     ax3d.plot3D(X[:, 0], X[:, 1], X[:, 2], "r-", linewidth=lw)
#     ax3d.plot3D(Y[:, 0], Y[:, 1], Y[:, 2], "g-", linewidth=lw)
#     ax3d.plot3D(Z[:, 0], Z[:, 1], Z[:, 2], "b-", linewidth=lw)


# @torch.no_grad()
# def draw(
#     ax3d,
#     graph: kinematics.SkeletonGraph,
#     anchors: torch.Tensor = None,
#     draw_local_frames: bool = True,
#     draw_vertex_labels: bool = False,
#     hide_root: bool = True,
# ):
#     fkt = kinematics.fk(graph)
#     root = graph.graph["root"]
#     for u, v in graph.graph["bfs_edges"]:
#         if u == root and hide_root:
#             continue
#         x = fkt[[u, v], 0, 3].numpy()
#         y = fkt[[u, v], 1, 3].numpy()
#         z = fkt[[u, v], 2, 3].numpy()
#         ax3d.plot(x, y, z, lw=1, c="k", linestyle="--")
#         ax3d.scatter(x[1], y[1], z[1], c="k")
#     if draw_local_frames:
#         if not hide_root:
#             draw_axis(ax3d, fkt[root])
#         for _, v in graph.graph["bfs_edges"]:
#             draw_axis(ax3d, fkt[v])
#     if draw_vertex_labels:
#         for u, label in graph.nodes.data("label"):
#             if u == root and hide_root:
#                 continue
#             xyz = fkt[u, :3, 3].numpy()
#             ax3d.text(xyz[0], xyz[1], xyz[2], label, fontsize="x-small")

#     if anchors is not None:
#         anchors = anchors.numpy()
#         ax3d.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], c="r", marker="o")
#         # for u, label in graph.nodes.data("label"):
#         #     ax3d.text(
#         #         anchors[u, 0], anchors[u, 1], anchors[u, 2], label, fontsize="x-small"
#         #     )


def main():
    g = kinematics.SkeletonGenerator()
    g.bone(
        "torso",
        "chest",
        utils.make_tuv(1.17965, "-x,z,y"),
        **utils.make_dofs(rx=0.0, irx=(-10.0, 90.0)),
    )
    g.bone(
        "chest",
        "neck",
        utils.make_tuv(2.0279, "x,y,z"),
        **utils.make_dofs(ry=0.0, iry=(-90.0, 90.0)),
    )
    g.bone("neck", "head", utils.make_tuv(0.73577, "-x,y,-z"))

    g.bone(
        "neck",
        "shoulder.L",
        utils.make_tuv(0.71612, "-z,-x,y"),
    )
    g.bone(
        "shoulder.L",
        "elbow.L",
        utils.make_tuv(1.8189, "x,y,z"),
        **utils.make_dofs(
            rx=0,
            irx=(-90.0, 30.0),
            ry=0.0,
            iry=(-90.0, 90.0),
            rz=0.0,
            irz=(-90.0, 90.0),
        ),
    )
    g.bone(
        "elbow.L",
        "hand.L",
        utils.make_tuv(1.1908, "x,y,z"),
        **utils.make_dofs(rz=0, irz=(-135.0, 0.0)),
    )

    g.bone("neck", "shoulder.R", utils.make_tuv(0.71612, "z,x,y"))
    g.bone(
        "shoulder.R",
        "elbow.R",
        utils.make_tuv(1.8189, "x,y,z"),
        **utils.make_dofs(
            rx=0,
            irx=(-90.0, 30.0),
            rz=0.0,
            irz=(-90.0, 90.0),
            ry=0.0,
            iry=(-90.0, 90.0),
        ),
    )
    g.bone(
        "elbow.R",
        "hand.R",
        utils.make_tuv(1.1908, "x,y,z"),
        **utils.make_dofs(rz=0, irz=(0.0, 135.0)),
    )

    g.bone("torso", "hip.L", utils.make_tuv(1.1542, "-y,x,z"))
    g.bone(
        "hip.L",
        "knee.L",
        utils.make_tuv(2.2245, "x,-z,y"),
        **utils.make_dofs(
            ry=0, iry=(-90.0, 90), rx=0.0, irx=(-20.0, 20), rz=0.0, irz=(-20.0, 20.0)
        ),
    )
    g.bone(
        "knee.L",
        "foot.L",
        utils.make_tuv(1.7149, "x,y,z"),
        **utils.make_dofs(rz=0, irz=(0.0, 90.0)),
    )

    g.bone("torso", "hip.R", utils.make_tuv(1.1542, "y,-x,z"))
    g.bone(
        "hip.R",
        "knee.R",
        utils.make_tuv(2.2245, "x,-z,y"),
        **utils.make_dofs(
            ry=0.0, iry=(-90.0, 90), rx=0.0, irx=(-20.0, 20), rz=0.0, irz=(-20.0, 20.0)
        ),
    )
    g.bone(
        "knee.R",
        "foot.R",
        utils.make_tuv(1.7149, "x,y,z"),
        **utils.make_dofs(rz=0, irz=(-90.0, 0.0)),
    )

    g.bone(
        "root",
        "torso",
        torch.eye(4),
        **utils.make_dofs(rx=0, ry=0, rz=0, tx=0, ty=0, tz=0),
    )

    graph = g.create_graph(
        [
            "head",
            "neck",
            "shoulder.R",
            "elbow.R",
            "hand.R",
            "shoulder.L",
            "elbow.L",
            "hand.L",
            "hip.R",
            "knee.R",
            "foot.R",
            "hip.L",
            "knee.L",
            "foot.L",
            "torso",
            "chest",
            "root",
        ]
    )

    N = graph.number_of_nodes()
    frame_data = pickle.load(open(r"C:\dev\bone-solve-ik\etc\frames.pkl", "rb"))

    poses = []
    # poses.append(kinematics.fk(graph))
    # graph[2][3]["bone"].rx.set_angle(-90)
    # poses.append(kinematics.fk(graph))

    solver = solvers.IKSolver(graph)
    anchors = torch.zeros((N, 3))
    weights = torch.ones(N)
    weights[-1] = 0

    axes_ranges = [[-20, 20], [-20, 20], [-2, 5]]

    # for i in range(400, 600, 10):
    for i in [510, 520]:
        print(f"Solving {i}", end=None)
        anchors[: N - 1] = torch.from_numpy(frame_data[i]).float()
        xyz = anchors[-3].clone()
        anchors[: N - 1] -= xyz
        loss = solver.solve(anchors, weights)
        if loss > 0.3:
            print(f", retry", end=None)
            kinematics.reset_dofs(graph)
            loss = solver.solve(anchors, weights)
        if loss < 0.1:
            print(f", solved", end=None)
            # graph[16][14]["bone"].tx.set_offset(xyz[0])
            # graph[16][14]["bone"].ty.set_offset(xyz[1])
            # graph[16][14]["bone"].tz.set_offset(xyz[2])
            poses.append(kinematics.fk(graph))
        # anchors[: N - 1] += xyz

        fig, ax = draw.create_figure3d(axes_ranges=axes_ranges)
        draw.draw(ax, graph, anchors, draw_vertex_labels=False, draw_local_frames=False)
        fig.savefig(f"tmp/{i:05d}.png", bbox_inches="tight")
        plt.show(block=True)
        # plt.pause(0.01)

        if loss >= 0.1:
            kinematics.reset_dofs(graph)
        print()

    bvh.export_bvh(graph, poses, frame_time=1 / 3.0)


def makefile():
    # ffmpeg -f concat -i tmp\concat.txt -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p skel.mp4
    from glob import glob
    from pathlib import Path

    files = sorted(glob("tmp/*.png"))
    with open("tmp/concat.txt", "w") as f:
        f.writelines([f"file {Path(f).name}\nduration 0.25\n" for f in files])


if __name__ == "__main__":
    main()
    # makefile()
