from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn
import networkx as nx

from .dofs import RotX, RotY, RotZ


class Bone(torch.nn.Module):
    """Edge data in the skeleton graph."""

    def __init__(
        self,
        t_uv: torch.tensor = None,
        rotx: Optional[RotX] = None,
        roty: Optional[RotY] = None,
        rotz: Optional[RotZ] = None,
    ) -> None:
        super().__init__()
        if t_uv is None:
            t_uv = torch.eye(4)
        if rotx is None:
            rotx = RotX(unlocked=False)
        if roty is None:
            roty = RotY(unlocked=False)
        if rotz is None:
            rotz = RotZ(unlocked=False)
        self.t_uv = torch.as_tensor(t_uv).float()
        self.rotx = rotx
        self.roty = roty
        self.rotz = rotz

    def matrix(self) -> torch.Tensor:
        # This should be interpreted as follows: the transformation from v to u is
        # given by first transforming wrt node u, using the rest-transformation, and
        # then applying a sequence of rotations around node u.
        return self.rotx.matrix() @ self.roty.matrix() @ self.rotz.matrix() @ self.t_uv

    def project_(self):
        self.rotx.project_()
        self.roty.project_()
        self.rotz.project_()

    def __str__(self):
        return f"Bone(rotx={self.rotx.angle:.3f},roty={self.roty.angle:.3f},rotz={self.rotz.angle:.3f})"


Vertex = Any
Edge = Tuple[Vertex, Vertex]
SkeletonGraph = nx.DiGraph
VertexTensorDict = Dict[Vertex, torch.Tensor]


class SkeletonGenerator:
    def __init__(
        self,
    ) -> None:
        self.graph = nx.DiGraph()

    def bone(
        self,
        u: Vertex,
        v: Vertex,
        t_uv: torch.tensor = None,
        rotx: Optional[RotX] = None,
        roty: Optional[RotY] = None,
        rotz: Optional[RotZ] = None,
    ) -> "SkeletonGenerator":
        self.graph.add_edge(u, v, bone=Bone(t_uv, rotx, roty, rotz))
        return self

    def create_graph(self) -> SkeletonGraph:
        root = [n for n, d in self.graph.in_degree() if d == 0][0]
        self.graph.graph["bfs_edges"] = list(nx.bfs_edges(self.graph, root))
        self.graph.graph["root"] = root
        return self.graph


def fk(graph: SkeletonGraph) -> VertexTensorDict:
    """Computes the forward kinematic poses of each vertex."""
    fk_dict = {}
    fk_dict[graph.graph["root"]] = torch.eye(4)
    for u, v in graph.graph["bfs_edges"]:
        bone: Bone = graph[u][v]["bone"]
        fk_dict[v] = fk_dict[u] @ bone.matrix()
    return fk_dict


def fmt_skeleton(graph: SkeletonGraph):
    parts = []
    max_node_width = 0
    for n in graph.nodes:
        max_node_width = max(len(str(n)), max_node_width)
    fmt_str = f"{{u:>{max_node_width}s}} -> {{v:{max_node_width}s}} : {{bone}}"
    for u, v in graph.graph["bfs_edges"]:
        bone: Bone = graph[u][v]["bone"]
        parts.append(fmt_str.format(u=str(u), v=str(v), bone=bone))
    return "\n".join(parts)
