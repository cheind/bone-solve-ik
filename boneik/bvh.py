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
            succ = list(graph.successors(v))
            if len(succ) == 0:
                succ = [None]  # will become an end-site
            for idx, n in enumerate(succ):
                _traverse((v, n), e, depth + 1, idx)
        lines.extend(_end_joint(depth, intend))

    _traverse(graph.graph["bfs_edges"][0], (None, root), 0, 0)
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
    root = graph.graph["root"]
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
