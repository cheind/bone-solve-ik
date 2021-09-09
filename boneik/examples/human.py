from boneik.dofs import RotX
from boneik.reparametrizations import PI
import matplotlib.pyplot as plt
import transformations as T
import numpy as np
import torch
import pickle
from tqdm import tqdm
from boneik import kinematics, solvers, utils, draw, criteria
from boneik import bvh


def main():
    b = kinematics.KinematicBuilder()
    b.add_bone(
        "torso",
        "chest",
        tip_to_base=utils.make_tip_to_base(1.17965, "-x,z,y"),
        dofs={"rx": np.deg2rad([-10.0, 90.0])},
    ).add_bone(
        "chest",
        "neck",
        tip_to_base=utils.make_tip_to_base(2.0279, "x,y,z"),
        dofs={"ry": np.deg2rad([-90.0, 90.0])},
    ).add_bone(
        "neck",
        "head",
        tip_to_base=utils.make_tip_to_base(0.73577, "-x,y,-z"),
    ).add_bone(
        "neck",
        "shoulder.L",
        tip_to_base=utils.make_tip_to_base(0.71612, "-z,-x,y"),
    ).add_bone(
        "shoulder.L",
        "elbow.L",
        tip_to_base=utils.make_tip_to_base(1.8189, "x,y,z"),
        dofs={
            "rx": np.deg2rad([-90.0, 30.0]),
            "ry": np.deg2rad([-90.0, 90.0]),
            "rz": np.deg2rad([-90.0, 90.0]),
        },
    ).add_bone(
        "elbow.L",
        "hand.L",
        tip_to_base=utils.make_tip_to_base(1.1908, "x,y,z"),
        dofs={"rz": np.deg2rad([-135.0, 0.0])},
    ).add_bone(
        "neck",
        "shoulder.R",
        tip_to_base=utils.make_tip_to_base(0.71612, "z,x,y"),
    ).add_bone(
        "shoulder.R",
        "elbow.R",
        tip_to_base=utils.make_tip_to_base(1.8189, "x,y,z"),
        dofs={
            "rx": np.deg2rad([-90.0, 30.0]),
            "ry": np.deg2rad([-90.0, 90.0]),
            "rz": np.deg2rad([-90.0, 90.0]),
        },
    ).add_bone(
        "elbow.R",
        "hand.R",
        tip_to_base=utils.make_tip_to_base(1.1908, "x,y,z"),
        dofs={"rz": np.deg2rad([0.0, 135.0])},
    ).add_bone(
        "torso",
        "hip.L",
        tip_to_base=utils.make_tip_to_base(1.1542, "-y,x,z"),
    ).add_bone(
        "hip.L",
        "knee.L",
        tip_to_base=utils.make_tip_to_base(2.2245, "x,-z,y"),
        dofs={
            "rx": np.deg2rad([-20.0, 20.0]),
            "ry": np.deg2rad([-90.0, 90.0]),
            "rz": np.deg2rad([-20.0, 20.0]),
        },
    ).add_bone(
        "knee.L",
        "foot.L",
        tip_to_base=utils.make_tip_to_base(1.7149, "x,y,z"),
        dofs={"rz": np.deg2rad([0.0, 90.0])},
    ).add_bone(
        "torso",
        "hip.R",
        tip_to_base=utils.make_tip_to_base(1.1542, "y,-x,z"),
    ).add_bone(
        "hip.R",
        "knee.R",
        tip_to_base=utils.make_tip_to_base(2.2245, "x,-z,y"),
        dofs={
            "rx": np.deg2rad([-20.0, 20.0]),
            "ry": np.deg2rad([-90.0, 90.0]),
            "rz": np.deg2rad([-20.0, 20.0]),
        },
    ).add_bone(
        "knee.R",
        "foot.R",
        tip_to_base=utils.make_tip_to_base(1.7149, "x,y,z"),
        dofs={"rz": np.deg2rad([-90.0, 0.0])},
    ).add_bone(
        "root",
        "torso",
        tip_to_base=torch.eye(4),
        dofs={"rx", "ry", "rz", "tx", "ty", "tz"},
    )

    kinematic = b.finalize(
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

    N = kinematic.graph.number_of_nodes()
    frame_data = pickle.load(open(r"C:\dev\bone-solve-ik\etc\frames.pkl", "rb"))

    poses = [kinematic.fk()]  # important to start from rest-pose for bvh export.

    solver = solvers.IKSolver(kinematic)
    crit = criteria.EuclideanDistanceCriterium(torch.zeros((N, 3)), torch.ones(N))
    crit.weights[-1] = 0  # root joint does not have any anchor.

    axes_ranges = [[-20, 20], [-20, 20], [-2, 5]]
    fig, ax = draw.create_figure3d(axes_ranges=axes_ranges)

    prev_pose = kinematic.fk()
    for i in tqdm(range(0, len(frame_data), 5)):
        crit.anchors[: N - 1] = torch.from_numpy(frame_data[i]).float()
        torso = crit.anchors[-3].clone()
        crit.anchors[: N - 1] -= torso  # torso at 0/0/0
        loss = solver.solve(crit, history_size=10, max_iter=10)
        if loss > 0.3:
            # retry from rest-pose
            kinematic.reset_()
            loss = solver.solve(crit)
        if loss < 0.3:
            # print("+", end="")
            delta = kinematic["root", "torso"].get_delta()
            kinematic["root", "torso"].set_delta(
                [
                    delta[0],
                    delta[1],
                    delta[2],
                    delta[3] + torso[0],
                    delta[4] + torso[1],
                    delta[5] + torso[2],
                ]
            )
            new_pose = kinematic.fk()
            poses.append(new_pose)
            prev_pose = new_pose
        else:
            kinematic.reset_()
            poses.append(prev_pose)  # Do not skip any frames, unhandled by BVH
        crit.anchors[: N - 1] += torso
        ax.cla()
        ax.set_xlim(*axes_ranges[0])
        ax.set_ylim(*axes_ranges[1])
        ax.set_zlim(*axes_ranges[2])
        draw.draw_kinematics(
            ax,
            kinematic=kinematic,
            fk=kinematic.fk(),
            anchors=crit.anchors,
            draw_vertex_labels=False,
            draw_local_frames=False,
            draw_root=False,
        )
        # fig.savefig(f"tmp/{i:05d}.png", bbox_inches="tight")
        plt.show(block=False)
        plt.pause(0.01)

    bvh.export_bvh(path="tmp/human.bvh", kinematic=kinematic, poses=poses, fps=1 / 3.0)


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
