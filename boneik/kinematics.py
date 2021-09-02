from typing import Dict, Optional, Tuple, Any, List

import torch
import torch.nn
import networkx as nx

from .dofs import RotX, RotY, RotZ, TransX, TransY, TransZ


class Bone(torch.nn.Module):
    """Edge data in the skeleton graph."""

    def __init__(
        self,
        t_uv: torch.tensor = None,
        rx: Optional[RotX] = None,
        ry: Optional[RotY] = None,
        rz: Optional[RotZ] = None,
        tx: Optional[TransX] = None,
        ty: Optional[TransY] = None,
        tz: Optional[TransZ] = None,
    ) -> None:
        super().__init__()
        if t_uv is None:
            t_uv = torch.eye(4)
        if rx is None:
            rx = RotX(unlocked=False)
        if ry is None:
            ry = RotY(unlocked=False)
        if rz is None:
            rz = RotZ(unlocked=False)
        if tx is None:
            tx = TransX(unlocked=False)
        if ty is None:
            ty = TransY(unlocked=False)
        if tz is None:
            tz = TransZ(unlocked=False)
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.t_uv = torch.as_tensor(t_uv).float()

    def matrix(self) -> torch.Tensor:
        # This should be interpreted as follows: the transformation from v to u is
        # given by first transforming wrt node u, using the rest-transformation, and
        # then applying a sequence of rotations around node u.
        return (
            self.tx.matrix()
            @ self.ty.matrix()
            @ self.tz.matrix()
            @ self.rx.matrix()
            @ self.ry.matrix()
            @ self.rz.matrix()
            @ self.t_uv
        )

    def project_(self):
        self.rx.project_()
        self.ry.project_()
        self.rz.project_()

    def reset_(self):
        self.rx.reset_()
        self.ry.reset_()
        self.rz.reset_()
        self.tx.reset_()
        self.ty.reset_()
        self.tz.reset_()

    def __str__(self):
        return (
            "Bone(r=["
            f"{self.rx.angle.item():.3f},{self.ry.angle.item():.3f},{self.rz.angle.item():.3f}"
            "], t=["
            f"{self.tx.offset.item():.3f},{self.ty.offset.item():.3f},{self.tz.offset.item():.3f}"
            "])"
        )


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
        t_uv: torch.Tensor = None,
        rx: Optional[RotX] = None,
        ry: Optional[RotY] = None,
        rz: Optional[RotZ] = None,
        tx: Optional[TransX] = None,
        ty: Optional[TransY] = None,
        tz: Optional[TransZ] = None,
    ) -> "SkeletonGenerator":
        self.graph.add_edge(u, v, bone=Bone(t_uv, rx, ry, rz, tx, ty, tz))
        return self

    def create_graph(self, relabel_order: List[Vertex] = None) -> SkeletonGraph:
        """Returns the final skeleton graph with nodes relabled in range [0,N)."""
        N = self.graph.number_of_nodes()
        if relabel_order is None:
            relabel_order = list(self.graph.nodes())
        node_mapping = dict(zip(relabel_order, range(N)))
        graph = nx.relabel_nodes(self.graph, node_mapping, copy=False)
        nx.set_node_attributes(graph, {v: k for k, v in node_mapping.items()}, "label")
        roots = [n for n, d in graph.in_degree() if d == 0]
        if len(roots) > 1:
            raise ValueError("More than one skeleton root.")
        graph.graph["bfs_edges"] = list(nx.bfs_edges(graph, roots[0]))
        graph.graph["root"] = roots[0]
        return graph


def fk(graph: SkeletonGraph) -> torch.FloatTensor:
    """Computes the forward kinematic poses of each vertex."""
    N = graph.number_of_nodes()
    root = graph.graph["root"]
    fkt = [torch.zeros((4, 4)) for _ in range(N)]
    fkt[root] = torch.eye(4)
    for u, v in graph.graph["bfs_edges"]:
        bone: Bone = graph[u][v]["bone"]
        fkt[v] = fkt[u] @ bone.matrix()
    return torch.stack(fkt, 0)


def reset_dofs(graph: SkeletonGraph):
    for u, v in graph.graph["bfs_edges"]:
        bone: Bone = graph[u][v]["bone"]
        bone.reset_()


def fmt_skeleton(graph: SkeletonGraph):
    parts = []
    max_node_width = 0
    for n in graph.nodes:
        max_node_width = max(len(str(n)), max_node_width)
    fmt_str = f"{{u:>{max_node_width}s}} -> {{v:{max_node_width}s}} : {{bone}}"
    for u, v in graph.graph["bfs_edges"]:
        bone: Bone = graph[u][v]["bone"]
        ulabel = graph.nodes[u]["label"]
        vlabel = graph.nodes[v]["label"]
        parts.append(fmt_str.format(u=str(ulabel), v=str(vlabel), bone=bone))
    return "\n".join(parts)
