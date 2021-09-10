from typing import Dict, Optional, Tuple, Any, List, Union

import torch
import torch.nn
import networkx as nx

from . import bones

Vertex = Any
Edge = Tuple[Vertex, Vertex]
VertexTensorDict = Dict[Vertex, torch.Tensor]


class Body:
    def __init__(
        self, graph: nx.DiGraph, root: Vertex, node_mapping: Dict[Vertex, int]
    ) -> None:
        self.graph = graph
        self.root = root
        self.node_mapping = node_mapping
        self.bfs_edges = list(nx.bfs_edges(graph, root))

    def fk(self) -> torch.FloatTensor:
        """Computes the forward kinematic poses of each vertex."""
        N = self.graph.number_of_nodes()
        fkt = [torch.empty((4, 4)) for _ in range(N)]
        fkt[self.root] = torch.eye(4)
        for u, v in self.bfs_edges:
            bone: bones.Bone = self.graph[u][v]["bone"]
            fkt[v] = fkt[u] @ bone.matrix()
        return torch.stack(fkt, 0)

    def project_(self):
        for _, _, bone in self.graph.edges(data="bone"):
            bone: bones.Bone
            bone.project_()

    def reset_(self):
        for _, _, bone in self.graph.edges(data="bone"):
            bone: bones.Bone
            bone.reset_()

    def __str__(self):
        parts = []
        max_node_width = 0
        for n in self.graph.nodes:
            max_node_width = max(len(str(n)), max_node_width)
        fmt_str = f"{{u:>{max_node_width}s}} -> {{v:{max_node_width}s}} : {{bone}}"
        for u, v in self.bfs_edges:
            bone: bones.Bone = self.graph[u][v]["bone"]
            ulabel = self.graph.nodes[u]["label"]
            vlabel = self.graph.nodes[v]["label"]
            parts.append(fmt_str.format(u=str(ulabel), v=str(vlabel), bone=bone))
        return "\n".join(parts)

    def __getitem__(self, uv: Tuple[Vertex, Vertex]) -> bones.Bone:
        u, v = uv
        return self.graph[self.node_mapping[u]][self.node_mapping[v]]["bone"]


class BodyBuilder:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def add_bone(
        self,
        u: Vertex,
        v: Vertex,
        *,
        tip_to_base: torch.FloatTensor = None,
        dofs: Union[bones.DofDict, bones.DofSet] = None,
    ) -> "BodyBuilder":
        if tip_to_base is None:
            tip_to_base = torch.eye(4)
        if isinstance(dofs, set):
            dofs = {d: None for d in dofs}
        self.graph.add_edge(u, v, bone=bones.Bone(tip_to_base, dof_dict=dofs))
        return self

    def finalize(self, relabel_order: List[Vertex] = None) -> Body:
        """Returns the final kinematic graph with nodes relabled in range [0,N)."""
        N = self.graph.number_of_nodes()
        if relabel_order is None:
            relabel_order = list(self.graph.nodes())
        else:
            assert len(set(relabel_order)) == N
        node_mapping = dict(zip(relabel_order, range(N)))
        graph = nx.relabel_nodes(self.graph, node_mapping, copy=False)
        nx.set_node_attributes(graph, {v: k for k, v in node_mapping.items()}, "label")
        roots = [n for n, d in graph.in_degree() if d == 0]
        if len(roots) > 1:
            raise ValueError("More than one skeleton root.")

        return Body(self.graph, roots[0], node_mapping)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from .utils import make_tip_to_base
    from . import draw

    b = BodyBuilder()
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
    body = b.finalize(["root", "a", "b", "c", "d", "e"])

    poses = [body.fk()]

    body["root", "a"].set_delta([np.pi / 4, 0, 0, 2.0, 0, 0])
    body["c", "d"].set_delta([0, 0, -np.pi / 2, 0, 0, 0])
    body["c", "e"].set_delta([0, 0, -np.pi / 2, 0, 0, 0])
    poses.append(body.fk())

    fig = plt.figure()
    ax0 = draw.create_axis3d(
        fig, pos=(1, 1, 1), axes_ranges=[[-2, 2], [-2, 2], [-2, 2]]
    )
    draw.draw_kinematics(
        ax0,
        body=body,
        fk=poses[0],
        draw_vertex_labels=True,
        draw_local_frames=True,
        draw_root=True,
    )
    draw.draw_kinematics(
        ax0,
        body=body,
        fk=poses[1],
        draw_vertex_labels=True,
        draw_local_frames=True,
        draw_root=True,
    )
    plt.show()