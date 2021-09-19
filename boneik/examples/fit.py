import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
from tqdm import tqdm
from boneik import bodies, ik_solvers, utils, draw, ik_criteria, io
from boneik import bvh


def create_human_body() -> bodies.Body:
    b = bodies.BodyBuilder()
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
        # dofs={"rx", "ry", "rz", "tx", "ty", "tz"},
        dofs={"rx", "ry", "rz"},
    )

    body = b.finalize(
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

    return body


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Pickled 3D joint predictions (NxMx3)")
    parser.add_argument("-body", type=Path, help="Kinematic description file")
    parser.add_argument("-input-fps", type=int, default=30, help="Input FPS")
    parser.add_argument("-input-step", type=int, default=1, help="Fit every nth frame")
    parser.add_argument(
        "-scale", type=float, help="Scale anchors of first frame to this"
    )
    parser.add_argument(
        "-max-loss", type=float, default=0.3, help="max loss to accept in fitting"
    )
    parser.add_argument(
        "-crit",
        type=str,
        choices=["euclidean", "parallel"],
        default="parallel",
        help="Loss criterium to apply",
    )
    parser.add_argument("-output", type=Path, default=Path("./tmp/human.bvh"))
    parser.add_argument("-show", type=int, default=1, help="visualize every nth frame")

    args = parser.parse_args()
    assert args.input.is_file()

    if args.body is not None:
        assert args.body.is_file()
        body = io.load_json(args.body)
    else:
        body = create_human_body()
    kin = body.kinematics()
    N = body.graph.number_of_nodes()
    frame_data = pickle.load(open(args.input, "rb"))
    if args.scale is not None:
        scale_factor = utils.find_scale_factor(frame_data[0]) * args.scale
    else:
        scale_factor = 1.0

    log_angles = kin.log_angles_rest_pose().requires_grad_(True)

    poses = [
        kin.fk(log_angles.detach())[0]
    ]  # important to start from rest-pose for bvh export.

    if args.crit == "parallel":
        crit = ik_criteria.ParallelSegmentCriterium(torch.zeros((N, 3)), torch.ones(N))
    else:
        crit = ik_criteria.EuclideanDistanceCriterium(
            torch.zeros((N, 3)), torch.ones(N)
        )
    crit.weights[-1] = 0  # root joint never has a corresponding anchor.

    axes_ranges = [[-20, 20], [-20, 20], [-2, 5]]
    fig, ax = draw.create_figure3d(axes_ranges=axes_ranges)
    prev_pose = poses[0]
    for i in tqdm(range(0, len(frame_data), args.input_step)):
        crit.anchors[: N - 1] = torch.from_numpy(frame_data[i]).float() * scale_factor
        torso = crit.anchors[-3].clone()
        crit.anchors[: N - 1] -= torso  # torso at 0/0/0
        # Attempt to solve from prev. angles
        loss = ik_solvers.solve_ik(kin, log_angles, crit, history_size=10, max_iter=10)
        if loss > args.max_loss:
            # retry from rest-pose
            log_angles = kin.log_angles_rest_pose().requires_grad_(True)
            loss = ik_solvers.solve_ik(kin, log_angles, crit)
        if loss < args.max_loss:
            new_pose = kin.fk(log_angles.detach())[0]
            new_pose[:, :3, 3] += torso.view(1, 3)
            poses.append(new_pose)
            prev_pose = new_pose
        else:
            body.reset_()
            poses.append(prev_pose)  # Do not skip any frames, unhandled by BVH
        crit.anchors[: N - 1] += torso
        if (i // args.input_step) % args.show == 0:
            ax.cla()
            ax.set_xlim(*axes_ranges[0])
            ax.set_ylim(*axes_ranges[1])
            ax.set_zlim(*axes_ranges[2])
            draw.draw_kinematics(
                ax,
                body=body,
                fk=poses[-1],
                anchors=crit.anchors,
                draw_vertex_labels=False,
                draw_local_frames=False,
                draw_root=False,
            )
            # fig.savefig(f"tmp/{i:05d}.png", bbox_inches="tight")
            plt.show(block=False)
            plt.pause(0.01)

    bvh.export_bvh(
        path=args.output, body=body, poses=poses, fps=(args.input_fps / args.input_step)
    )


if __name__ == "__main__":
    main()
    # makefile()
