from boneik.dofs import RotX
from boneik.reparametrizations import PI
import matplotlib.pyplot as plt
import transformations as T
import numpy as np
import torch
import pickle
from tqdm import tqdm
from boneik import kinematics, solvers, utils, draw
from boneik import bvh


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

    poses = [kinematics.fk(graph)]  # important to start here for BVH

    solver = solvers.IKSolver(graph)
    anchors = torch.zeros((N, 3))
    weights = torch.ones(N)
    weights[-1] = 0

    axes_ranges = [[-20, 20], [-20, 20], [-2, 5]]
    fig, ax = draw.create_figure3d(axes_ranges=axes_ranges)

    for i in tqdm(range(0, len(frame_data), 10)):
        # for i in [200, 510, 515, 520]:
        anchors[: N - 1] = torch.from_numpy(frame_data[i]).float()
        xyz = anchors[-3].clone()
        anchors[: N - 1] -= xyz
        loss = solver.solve(anchors, weights)
        if loss > 0.3:
            kinematics.reset_dofs(graph)
            loss = solver.solve(anchors, weights)
        if loss < 0.3:
            # print("+", end="")
            graph[16][14]["bone"].tx.set_offset(xyz[0])
            graph[16][14]["bone"].ty.set_offset(xyz[1])
            graph[16][14]["bone"].tz.set_offset(xyz[2])
            poses.append(kinematics.fk(graph))
        else:
            kinematics.reset_dofs(graph)
        anchors[: N - 1] += xyz

        ax.cla()
        ax.set_xlim(*axes_ranges[0])
        ax.set_ylim(*axes_ranges[1])
        ax.set_zlim(*axes_ranges[2])
        draw.draw_kinematics(
            ax,
            graph,
            fk=kinematics.fk(graph),
            anchors=anchors,
            draw_vertex_labels=False,
            draw_local_frames=False,
            draw_root=False,
        )
        # fig.savefig(f"tmp/{i:05d}.png", bbox_inches="tight")
        plt.show(block=False)
        plt.pause(0.01)

    bvh.export_bvh(graph, poses, fps=1 / 3.0)


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
