import torch
import transformations as T
from typing import List, Tuple
import numpy as np


from . import kinematics


def _begin_joint(
    jtype: str, name: str, off: torch.FloatTensor, dof: int, depth: int, intend: int
) -> List[str]:
    lines = []
    spaces = " " * depth * intend
    lines.append(f"{spaces}{jtype} {name}")
    lines.append(f"{spaces}{{")
    spaces = " " * (depth + 1) * intend
    lines.append(f"{spaces}OFFSET {off[0]:.4f} {off[1]:.4f} {off[2]:.4f}")
    if dof == 6:
        lines.append(
            f"{spaces}CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"
        )
    elif dof == 3:
        lines.append(f"{spaces}CHANNELS 3 Xrotation Yrotation Zrotation")
    return lines


def _end_joint(depth: int, intend: int) -> List[str]:
    spaces = " " * depth * intend
    return [f"{spaces}}}"]


def _generate_hierarchy(
    kinematic: kinematics.Kinematic,
    fk: torch.FloatTensor,
    intend: int = 4,
) -> Tuple[List[str], List[Tuple[int, int, int]]]:

    # Walks along the edges in dfs and creates joints along the way.
    # For each edge, the head (not the tip) is created as joint. The
    # offset is computed from the prev-edge source. Then all successors
    # of the target node are traversed. Each one, will become its one
    # joint, so that in blender a separate bone between edges that share
    # a common source. Otherwise, we only have one bone for all and thus
    # limited degrees of freedom. Note, for edges without successors, artificial
    # edges are created to enable end-sites to be added to the hierarchy.
    #
    # Joint a
    # {
    #   Joint b
    #   {
    #       Offset x y z; vector a->b in global coordinates, defines root pose locations
    #       Rotation; refers to how the bone starting at b is rotated wrt. to root pose in
    #                 global coordinates (subtracting the effect of previous members of the
    #                 hierarchy)
    #   }
    # }
    #

    lines = []
    motion_order = []
    root = kinematic.root
    graph = kinematic.graph
    bfs_edges = kinematic.bfs_edges

    def _traverse(
        e: Tuple[int, int], eprev: Tuple[int, int], depth: int, nthchild: int
    ):
        u, v = e
        w, _ = eprev
        ulabel = graph.nodes[u]["label"]
        label = f"{ulabel}.{nthchild:02d}"

        is_first = u == root
        is_endsite = v is None

        if is_endsite:
            off = (fk[u] - fk[w])[:3, 3]
            lines.extend(_begin_joint("End Site", label, off, 0, depth, intend))
        else:
            if is_first:
                lines.extend(_begin_joint("ROOT", label, [0, 0, 0], 6, depth, intend))
            else:
                off = (fk[u] - fk[w])[:3, 3]
                lines.extend(_begin_joint("JOINT", label, off, 3, depth, intend))
            motion_order.append(e)
            # Note the sorted. Otherwise we might generate bones with different
            # name suffixes
            succ = sorted(list(graph.successors(v)))
            if len(succ) == 0:
                succ = [None]  # will become an end-site
            for idx, n in enumerate(succ):
                _traverse((v, n), e, depth + 1, idx)
        lines.extend(_end_joint(depth, intend))

    _traverse(bfs_edges[0], (None, root), 0, 0)
    return lines, motion_order


def _rinv(m: torch.FloatTensor) -> torch.FloatTensor:
    minv = torch.eye(4)
    minv[:3, :3] = m[:3, :3].T
    minv[:3, 3] = -(m[:3, :3].T @ m[:3, 3])
    return minv


def _skew(x: torch.FloatTensor) -> torch.FloatTensor:
    m = torch.zeros((3, 3))
    m[0, 1] = -x[2]
    m[0, 2] = x[1]
    m[1, 0] = x[2]
    m[1, 2] = -x[0]
    m[2, 0] = -x[1]
    m[2, 1] = x[0]
    return m


def _rot(t: torch.FloatTensor, p: torch.FloatTensor) -> torch.FloatTensor:
    n = torch.cross(t, p)
    nn = torch.norm(n)
    if nn < 1e-6:
        R = torch.eye(3)
    else:
        s = _skew(n / nn)
        ntp = torch.norm(t) * torch.norm(p)
        cosalpha = torch.dot(t, p) / ntp
        sinalpha = nn / ntp
        R = torch.eye(3) + sinalpha * s + (1 - cosalpha) * torch.matrix_power(s, 2)
    return R


def _generate_motion(
    kinematic: kinematics.Kinematic,
    poses: List[torch.FloatTensor],
    motion_order: List[Tuple[int, int, int]],
    degrees: bool = True,
) -> List[str]:
    root = kinematic.root
    lines = []
    for fk in poses:
        parts = []
        for u, v in motion_order:
            if u == root:
                m = fk[v]
            else:
                tp = poses[0][u] @ _rinv(fk[u]) @ fk[v]
                t = poses[0][v][:3, 3] - poses[0][u][:3, 3]
                p = tp[:3, 3] - poses[0][u][:3, 3]
                R = _rot(t, p)
                m = torch.eye(4)
                m[:3, :3] = R
            rot = T.euler_from_matrix(m, axes="szyx")
            trans = T.translation_from_matrix(m)

            if degrees:
                rot = np.rad2deg(rot)
            if u == root:
                parts.append(
                    f"{trans[0]:.4f} {trans[1]:.4f} {trans[2]:.4f} {rot[2]:.4f} {rot[1]:.4f} {rot[0]:.4f}"
                )
            else:
                parts.append(f"{rot[2]:.4f} {rot[1]:.4f} {rot[0]:.4f}")

        lines.append(" ".join(parts))
    return lines


@torch.no_grad()
def create_bvh(
    kinematic: kinematics.Kinematic,
    poses: List[torch.FloatTensor],
    fps: float = 30,
    first_motion_frame: int = 0,
    rest_frame: int = 0,
    degrees: bool = True,
) -> str:
    lines = ["HIERARCHY"]
    motion_order = []
    hlines, motion_order = _generate_hierarchy(kinematic, fk=poses[rest_frame])
    lines.extend(hlines)
    lines.append("MOTION")
    lines.append(f"Frames: {len(poses)}")
    lines.append(f"Frame Time: {(1.0/fps):.4f}")
    mlines = _generate_motion(
        kinematic, poses[first_motion_frame:], motion_order, degrees=degrees
    )
    lines.extend(mlines)
    return "\n".join(lines)


@torch.no_grad()
def export_bvh(
    *,
    path: str,
    kinematic: kinematics.Kinematic,
    poses: List[torch.FloatTensor],
    fps: float = 30,
    first_motion_frame: int = 0,
    rest_frame: int = 0,
    degrees: bool = True,
) -> None:
    with open(path, "w") as f:
        f.write(
            create_bvh(
                kinematic,
                poses,
                fps=fps,
                first_motion_frame=first_motion_frame,
                rest_frame=rest_frame,
                degrees=degrees,
            )
        )


if __name__ == "__main__":

    from boneik import kinematics, draw
    from boneik.utils import make_tip_to_base
    import matplotlib.pyplot as plt

    b = kinematics.KinematicBuilder()
    b.add_bone(
        "a",
        "b",
        tip_to_base=make_tip_to_base(1.0, "x,y,z"),
        dofs={"rx": np.deg2rad([-90, 90])},
    )
    b.add_bone(
        "b",
        "c",
        tip_to_base=make_tip_to_base(0.5, "x,y,z"),
        dofs={"rx": np.deg2rad([-90, 90])},
    )
    b.add_bone(
        "c",
        "d",
        tip_to_base=make_tip_to_base(0.5, "z,-x,-y"),
        dofs={"rz": np.deg2rad([-90, 90])},
    )
    b.add_bone(
        "c",
        "e",
        tip_to_base=make_tip_to_base(0.5, "z,x,y"),
        dofs={"rz": np.deg2rad([-90, 90])},
    )
    b.add_bone("root", "a")
    k = b.finalize(["root", "a", "b", "c", "d", "e"])

    poses = [k.fk()]  # Rest pose

    k["root", "a"].set_delta([np.pi / 4, 0, 0, 2.0, 0, 0])
    k["c", "d"].set_delta([0, 0, -np.pi / 2, 0, 0, 0])
    k["c", "e"].set_delta([0, 0, np.pi / 2, 0, 0, 0])

    poses.append(k.fk())  # Final pose

    export_bvh("./tmp/robot.bvh", k, poses, fps=0.5)  # forward y, up z in blender

    fig = plt.figure()
    ax0 = draw.create_axis3d(
        fig, pos=(1, 2, 1), axes_ranges=[[-2, 2], [-2, 2], [-2, 2]]
    )
    ax0.set_title("Frame 0")
    ax1 = draw.create_axis3d(
        fig, pos=(1, 2, 2), axes_ranges=[[-2, 2], [-2, 2], [-2, 2]]
    )
    ax1.set_title("Frame 1")

    draw.draw_kinematics(
        ax0,
        kinematic=k,
        fk=poses[0],
        draw_vertex_labels=True,
        draw_local_frames=True,
        draw_root=True,
    )
    draw.draw_kinematics(
        ax1,
        kinematic=k,
        fk=poses[1],
        draw_vertex_labels=True,
        draw_local_frames=True,
        draw_root=True,
    )
    plt.show()
