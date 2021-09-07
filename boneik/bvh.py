from boneik.utils import make_dofs
from matplotlib.pyplot import axes
from networkx.classes.function import degree
from .kinematics import SkeletonGraph
import torch
import transformations as T
from typing import List, Tuple
import numpy as np
import networkx as nx


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
    graph: SkeletonGraph, fk: torch.FloatTensor, intend: int = 4
) -> Tuple[List[str], List[Tuple[int, int, int]]]:
    root = graph.graph["root"]
    lines = []
    motion_order = []

    def _traverse(u: int, parent: int, depth: int):
        ul = graph.nodes[u]["label"]
        succ = list(graph.successors(u))
        is_first = parent == root
        is_endsite = len(succ) == 0
        off = (fk[u] - fk[parent])[:3, 3]

        if is_endsite:
            lines.extend(_begin_joint("End Site", ul, off, 0, depth, intend))
        elif is_first:
            lines.extend(_begin_joint("ROOT", ul, [0, 0, 0], 6, depth, intend))
        else:
            lines.extend(_begin_joint("JOINT", ul, off, 3, depth, intend))

        if not is_endsite:
            motion_order.append((u, parent))

        if len(succ) == 1:
            _traverse(succ[0], u, depth + 1)
        elif len(succ) > 1:
            for idx, n in enumerate(succ):
                lines.extend(
                    _begin_joint(
                        "JOINT", f"{ul}.{idx:02d}", [0, 0, 0], 3, depth + 1, intend
                    )
                )
                _traverse(n, u, depth + 2)
                lines.extend(_end_joint(depth + 1, intend))
        lines.extend(_end_joint(depth, intend))

    # motion_order.append((nroot, None))
    _traverse(next(graph.successors(root)), root, 0)
    print(motion_order)

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
    graph: SkeletonGraph,
    poses: List[torch.FloatTensor],
    motion_order: List[Tuple[int, int, int]],
    degrees: bool = True,
) -> List[str]:
    lines = []
    for fk in poses:
        parts = []
        for u, parent in motion_order:

            tp = poses[0][parent] @ _rinv(fk[parent]) @ fk[u]
            t = poses[0][u][:3, 3] - poses[0][parent][:3, 3]
            p = tp[:3, 3] - poses[0][parent][:3, 3]
            R = _rot(t, p)
            m = torch.eye(4)
            m[:3, :3] = R
            # t = [0, 0, 0]
            r = T.euler_from_matrix(m, axes="sxyz")

            if degrees:
                r = np.rad2deg(r)
            if parent == graph.graph["root"]:
                print("root")
                t = poses[0][parent]
                off = [0, 0, 0]
                parts.append(
                    f"{off[0]:.4f} {off[1]:.4f} {off[2]:.4f} {r[0]:.4f} {r[1]:.4f} {r[2]:.4f}"
                )
            else:
                print("otherwise")
                parts.append(f"{r[0]:.4f} {r[1]:.4f} {r[2]:.4f}")
        lines.append(" ".join(parts))
    return lines


@torch.no_grad()
def export_bvh(
    graph: SkeletonGraph,
    poses: List[torch.FloatTensor],
    frame_time: float = 1.0 / 30,
    degrees: bool = True,
) -> List[str]:
    lines = ["HIERARCHY"]
    motion_order = []
    hlines, motion_order = _generate_hierarchy(graph, poses[0])
    lines.extend(hlines)
    lines.append("MOTION")
    lines.append(f"Frames: {len(poses)}")
    lines.append(f"Frame Time: {frame_time:.4f}")
    mlines = _generate_motion(graph, poses, motion_order, degrees=degrees)
    lines.extend(mlines)
    with open("test.bvh", "w") as f:
        f.write("\n".join(lines))


def blender_test():
    from boneik import kinematics, utils, draw
    import matplotlib.pyplot as plt

    g = kinematics.SkeletonGenerator()
    g.bone("a", "b", utils.make_tuv(1.0, "x,z,-y"), **utils.make_dofs(rz=0, rx=0, ry=0))
    g.bone("b", "c", utils.make_tuv(0.5, "x,y,z"), **utils.make_dofs(rx=0, ry=0, rz=0))
    graph = g.create_graph(["a", "b", "c"])

    poses = [kinematics.fk(graph)]

    graph[0][1]["bone"].rz.set_angle(np.deg2rad(-45))
    graph[1][2]["bone"].rx.set_angle(np.deg2rad(-45))
    graph[1][2]["bone"].ry.set_angle(np.deg2rad(45))
    graph[1][2]["bone"].rz.set_angle(np.deg2rad(-90))
    poses.append(kinematics.fk(graph))

    export_bvh(graph, poses, frame_time=2.0)

    ranges = [[-5, 5], [-5, 5], [-5, 5]]
    fig = plt.figure()
    ax0 = draw.create_axis3d(fig, (1, 2, 1), ranges)
    ax1 = draw.create_axis3d(fig, (1, 2, 2), ranges)

    draw.draw(ax0, graph, fk=poses[0], draw_local_frames=True, hide_root=False)
    draw.draw(ax1, graph, fk=poses[1], draw_local_frames=True, hide_root=False)
    plt.show()


if __name__ == "__main__":
    blender_test()
    # from boneik import kinematics, utils, draw
    # import matplotlib.pyplot as plt

    # g = kinematics.SkeletonGenerator()
    # g.bone(
    #     "a",
    #     "b",
    #     utils.make_tuv(1.0, "x,y,z"),
    #     **utils.make_dofs(rx=0.0, irx=(-90.0, 90.0)),
    # )
    # g.bone(
    #     "b",
    #     "c",
    #     utils.make_tuv(0.5, "x,y,z"),
    #     **utils.make_dofs(rx=0.0, irx=(-90.0, 90.0)),
    # )
    # g.bone(
    #     "c",
    #     "d",
    #     utils.make_tuv(0.5, "z,-x,-y"),
    #     **utils.make_dofs(rz=0.0, irz=(-90.0, 90.0)),
    # )
    # g.bone(
    #     "c",
    #     "e",
    #     utils.make_tuv(0.5, "z,x,y"),
    #     **utils.make_dofs(rz=0, irz=(-90.0, 90.0)),
    # )
    # graph = g.create_graph(["a", "b", "c", "d", "e"])

    # poses = [kinematics.fk(graph)]

    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # ax = fig.add_subplot(1, 1, 1, projection="3d")
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.set_xlim(-5.0, 5.0)
    # ax.set_xlabel("x")
    # ax.set_ylim(-5.0, 5.0)
    # ax.set_ylabel("y")
    # ax.set_zlim(-5.0, 5.0)
    # ax.set_zlabel("z")
    # # draw.draw(
    # #     ax,
    # #     graph,
    # #     anchors=None,
    # #     draw_vertex_labels=True,
    # #     draw_local_frames=True,
    # #     hide_root=False,
    # # )
    # # plt.show()

    # graph[0][1]["bone"].rx.set_angle(np.pi / 4)
    # graph[0][1]["bone"].tx.set_offset(2.0)
    # graph[2][3]["bone"].rz.set_angle(-np.pi / 2)
    # graph[2][4]["bone"].rz.set_angle(np.pi / 2)
    # poses.append(kinematics.fk(graph))

    # draw.draw(
    #     ax,
    #     graph,
    #     anchors=None,
    #     draw_vertex_labels=True,
    #     draw_local_frames=True,
    #     hide_root=False,
    # )
    # plt.show()

    # export_bvh(graph, poses, frame_time=2.0)  # forward y, up z in blender
