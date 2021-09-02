from matplotlib.pyplot import axes
from .kinematics import SkeletonGraph
import torch
import transformations as T
from typing import List, Tuple
import numpy as np


def _generate_hierarchy(
    u: int,
    p: int,
    graph: SkeletonGraph,
    fk: torch.FloatTensor,
    intend: int,
    node_order: List[int],
) -> List[str]:
    lines = []

    if u == graph.graph["root"]:
        node_order.append(u)
        lines.append(f"ROOT {graph.nodes[u]['label']}")
        lines.append("{")
        intend += 4
        lines.append(" " * intend + "OFFSET 0 0 0")
        lines.append(
            " " * intend
            + "CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"
        )
    else:
        t = (fk[u] - fk[p])[:3, 3]
        if graph.out_degree(u) == 0:
            lines.append(" " * intend + "End Site")
            lines.append(" " * intend + "{")
            intend += 4
            lines.append(" " * intend + f"OFFSET {t[0]:.4f} {t[1]:.4f} {t[2]:.4f}")
        else:
            node_order.append(u)
            lines.append(" " * intend + f"JOINT {graph.nodes[u]['label']}")
            lines.append(" " * intend + "{")
            intend += 4
            lines.append(" " * intend + f"OFFSET {t[0]:.4f} {t[1]:.4f} {t[2]:.4f}")
            lines.append(" " * intend + "CHANNELS 3 Xrotation Yrotation Zrotation")

    for v in graph.successors(u):
        lines.extend(_generate_hierarchy(v, u, graph, fk, intend, node_order))

    intend -= 4
    lines.append(" " * intend + "}")
    return lines


def _generate_motion(
    graph: SkeletonGraph, motion: List[torch.FloatTensor], node_order: List[int]
) -> List[str]:
    lines = []
    for fk in motion:
        parts = []
        for u in node_order:
            m = fk[u].detach().numpy()
            t = T.translation_from_matrix(m)
            r = T.euler_from_matrix(m, axes="szyx")
            r = np.rad2deg(r)
            if u == graph.graph["root"]:
                parts.append(
                    f"{t[0]:.4f} {t[1]:.4f} {t[2]:.4f} {r[0]:.4f} {r[1]:.4f} {r[2]:.4f}"
                )
            else:
                parts.append(f"{r[2]:.4f} {r[1]:.4f} {r[0]:.4f}")
        lines.append(" ".join(parts))
    return lines


@torch.no_grad()
def export_bvh(
    graph: SkeletonGraph, motion: List[torch.FloatTensor], frame_time: float = 1.0
) -> List[str]:
    lines = ["HIERARCHY"]
    node_order = []
    hlines = _generate_hierarchy(
        graph.graph["root"], None, graph, motion[0], 0, node_order
    )
    lines.extend(hlines)
    lines.append("MOTION")
    lines.append(f"Frames: {len(motion)}")
    lines.append(f"Frame Time: {frame_time:.4f}")
    mlines = _generate_motion(graph, motion, node_order)
    lines.extend(mlines)
    with open("test.bvh", "w") as f:
        f.write("\n".join(lines))
