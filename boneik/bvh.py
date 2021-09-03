from matplotlib.pyplot import axes
from networkx.classes.function import degree
from .kinematics import SkeletonGraph
import torch
import transformations as T
from typing import List, Tuple
import numpy as np
import networkx as nx

# def _generate_hierarchy(
#     u: int,
#     p: int,
#     graph: SkeletonGraph,
#     fk: torch.FloatTensor,
#     intend: int,
#     node_order: List[int],
# ) -> List[str]:
#     lines = []

#     if u == graph.graph["root"]:
#         node_order.append(u)
#         lines.append(f"ROOT {graph.nodes[u]['label']}")
#         lines.append("{")
#         intend += 4
#         lines.append(" " * intend + "OFFSET 0 0 0")
#         lines.append(
#             " " * intend
#             + "CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"
#         )
#     else:
#         t = (fk[u] - fk[p])[:3, 3]
#         if graph.out_degree(u) == 0:
#             lines.append(" " * intend + "End Site")
#             lines.append(" " * intend + "{")
#             intend += 4
#             lines.append(" " * intend + f"OFFSET {t[0]:.4f} {t[1]:.4f} {t[2]:.4f}")
#         else:
#             node_order.append(u)
#             lines.append(" " * intend + f"JOINT {graph.nodes[u]['label']}")
#             lines.append(" " * intend + "{")
#             intend += 4
#             lines.append(" " * intend + f"OFFSET {t[0]:.4f} {t[1]:.4f} {t[2]:.4f}")
#             lines.append(" " * intend + "CHANNELS 3 Xrotation Yrotation Zrotation")

#     for v in graph.successors(u):
#         lines.extend(_generate_hierarchy(v, u, graph, fk, intend, node_order))

#     intend -= 4
#     lines.append(" " * intend + "}")
#     return lines


def _begin_joint(
    jtype: str, name: str, off: torch.FloatTensor, depth: int, intend: int
) -> List[str]:
    lines = []
    spaces = " " * depth * intend
    lines.append(f"{spaces}{jtype} {name}")
    lines.append(f"{spaces}{{")
    spaces = " " * (depth + 1) * intend
    lines.append(f"{spaces}OFFSET {off[0]:.4f} {off[1]:.4f} {off[2]:.4f}")
    if jtype == "ROOT":
        lines.append(
            f"{spaces}CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"
        )
    elif jtype == "JOINT":
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
        print(parent, u)
        ul = graph.nodes[u]["label"]
        if parent is not None:
            off = (fk[u] - fk[parent])[:3, 3]
        else:
            off = [0.0, 0.0, 0.0]
        has_dof = True
        if graph.out_degree(u) == 0:
            jtype = "End Site"
            has_dof = False
        elif depth == 0:
            jtype = "ROOT"
        else:
            jtype = "JOINT"

        lines.extend(_begin_joint(jtype, ul, off, depth, intend))
        for n in list(graph.successors(u)):
            if has_dof:
                motion_order.append((n, u))
            _traverse(n, u, depth + 1)
        lines.extend(_end_joint(depth, intend))

    # motion_order.append((nroot, None))
    _traverse(root, None, 0)
    print(motion_order)

    return lines, motion_order


def _rigid_inv(m: torch.FloatTensor) -> torch.FloatTensor:
    minv = torch.eye(4)
    minv[:3, :3] = m[:3, :3].T
    minv[:3, 3] = -(m[:3, :3].T @ m[:3, 3])
    return minv


def _generate_motion(
    graph: SkeletonGraph,
    poses: List[torch.FloatTensor],
    motion_order: List[Tuple[int, int, int]],
    degrees: bool = True,
) -> List[str]:
    lines = []
    for fk in poses:
        parts = []
        for mo, (u, parent) in enumerate(motion_order):
            m = fk[u] @ _rigid_inv(poses[0][u])
            mask = abs(m) < 1e-6
            m[mask] = 0.0
            t = T.translation_from_matrix(m)
            r = T.euler_from_matrix(m, axes="sxyz")
            if degrees:
                r = np.rad2deg(r)
            if parent == graph.graph["root"]:
                parts.append(
                    f"{t[0]:.4f} {t[1]:.4f} {t[2]:.4f} {r[0]:.4f} {r[1]:.4f} {r[2]:.4f}"
                )
            else:
                parts.append(f"{r[0]:.4f} {r[1]:.4f} {r[2]:.4f}")
        lines.append(" ".join(parts))
    return lines


@torch.no_grad()
def export_bvh(
    graph: SkeletonGraph,
    poses: List[torch.FloatTensor],
    frame_time: float = 1.0,
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
