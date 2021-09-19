from typing import Any, Dict, List, Tuple, Union, Optional, Set

import torch
import torch.nn

import networkx as nx
import torch
import torch.nn

from . import rotations as R

Vertex = Any
Edge = Tuple[Vertex, Vertex]
VertexTensorDict = Dict[Vertex, torch.Tensor]
RangeConstraint = Optional[Tuple[float, float]]
DofDict = Dict[str, RangeConstraint]
DofSet = Set[str]


class Bone(torch.nn.Module):
    """Describes the geometric relationship between two joints.

    The bone is conceptionally an edge between two vertices (base, tip) of the
    kinematic graph. It stores the how the tip frame is rotated/translated wrt.
    the base in the rest pose.

    There is be a bone instance for every connected pair of joints (vertices)
    in the kinematic graph (see kinematics.Body). We store the bone as attribute
    of the edge connecting these joints. This allows several bones to be
    represented starting from a common joint.
    """

    def __init__(self, tip_to_base: torch.FloatTensor, dof_dict: DofDict) -> None:
        super().__init__()
        self.tip_to_base = torch.as_tensor(tip_to_base).float()
        self.dof_dict = dof_dict

    def matrix(self) -> torch.FloatTensor:
        return self.tip_to_base

    def __str__(self):
        return "Bone(tip_to_base=\n" f"{self.tip_to_base}" ")"


class BodyKinematics(torch.nn.Module):
    def __init__(self, body: "Body") -> None:
        super().__init__()
        self.body = body
        self._init_kinematics()

    def _init_kinematics(self):
        rot_unlock_mask = []
        rot_constraints = []
        rot_inv_constraints = []
        rot_axes = []
        rot_ranges = []
        tip_to_base = []
        axes = {
            "rx": torch.tensor([1.0, 0.0, 0.0]),
            "ry": torch.tensor([0.0, 1.0, 0.0]),
            "rz": torch.tensor([0.0, 0.0, 1.0]),
        }
        for u, v in self.body.bfs_edges:
            b: Bone = self.body.graph[u][v]["bone"]
            tip_to_base.append(b.tip_to_base)
            for rstr in ["rx", "ry", "rz"]:
                unlocked = rstr in b.dof_dict
                rot_unlock_mask.append(unlocked)
                if unlocked:
                    angle_range = torch.as_tensor(b.dof_dict[rstr]).float()
                else:
                    angle_range = R.UNCONSTRAINED_RANGE
                rot_ranges.append(angle_range)
                c, cinv = R.affine_constraint(angle_range)
                rot_constraints.append(c)
                rot_inv_constraints.append(cinv)
                rot_axes.append(axes[rstr])

        self.register_buffer("rot_open_ranges", torch.stack(rot_ranges, 0))
        self.register_buffer("rot_unlock_mask", torch.tensor(rot_unlock_mask))
        self.register_buffer("rot_constraints", torch.stack(rot_constraints, 0))
        self.register_buffer("rot_inv_constraints", torch.stack(rot_inv_constraints, 0))
        self.register_buffer("rot_axes", torch.stack(rot_axes, 0))
        self.register_buffer("tip_to_base", torch.stack(tip_to_base, 0))

    def log_angles_rest_pose(self):
        N = self.rot_constraints.shape[0]
        dev = self.rot_axes.device
        dtype = self.rot_axes.dtype
        theta = torch.zeros((1, N), dtype=dtype, device=dev)
        return R.log_map_angle(
            R.clamp_angle(theta, self.rot_open_ranges),
            self.rot_inv_constraints,
        )

    def fk(self, log_angles: torch.FloatTensor) -> torch.FloatTensor:
        rots = R.exp_map(log_angles, self.rot_axes, self.rot_constraints)
        B = log_angles.shape[0]  # number of samples
        N = self.body.graph.number_of_nodes()

        fkt = [None] * N
        e = torch.eye(4, device=log_angles.device, dtype=log_angles.dtype)
        e = e.reshape((1, 4, 4))
        e = e.repeat((B, 1, 1))
        fkt[self.body.root] = e

        for idx, (u, v) in enumerate(self.body.bfs_edges):
            rx, ry, rz = (
                rots[:, idx * 3],
                rots[:, idx * 3 + 1],
                rots[:, idx * 3 + 2],
            )
            r = rx @ ry @ rz  # Bx3x3
            t = log_angles.new_zeros((B, 4, 4))
            t[:, :3, :3] = r
            t[:, 3, 3] = 1.0
            fkt[v] = fkt[u] @ t @ self.tip_to_base[idx : idx + 1]
        return torch.stack(fkt, 1)


class Body:
    def __init__(
        self, graph: nx.DiGraph, root: Vertex, node_mapping: Dict[Vertex, int]
    ) -> None:
        self.graph = graph
        self.root = root
        self.node_mapping = node_mapping
        self.bfs_edges = list(nx.bfs_edges(graph, root))

    def __str__(self):
        parts = []
        max_node_width = 0
        for n in self.graph.nodes:
            max_node_width = max(len(str(n)), max_node_width)
        fmt_str = f"{{u:>{max_node_width}s}} -> {{v:{max_node_width}s}} : {{bone}}"
        for u, v in self.bfs_edges:
            bone: Bone = self.graph[u][v]["bone"]
            ulabel = self.graph.nodes[u]["label"]
            vlabel = self.graph.nodes[v]["label"]
            parts.append(fmt_str.format(u=str(ulabel), v=str(vlabel), bone=bone))
        return "\n".join(parts)

    def __getitem__(self, uv: Tuple[Vertex, Vertex]) -> Bone:
        u, v = uv
        return self.graph[self.node_mapping[u]][self.node_mapping[v]]["bone"]

    def kinematics(self) -> BodyKinematics:
        return BodyKinematics(self)


class BodyBuilder:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def add_bone(
        self,
        u: Vertex,
        v: Vertex,
        *,
        tip_to_base: torch.FloatTensor = None,
        dofs: Union[DofDict, DofSet] = None,
    ) -> "BodyBuilder":
        if tip_to_base is None:
            tip_to_base = torch.eye(4)
        if isinstance(dofs, set):
            dofs = {d: R.UNCONSTRAINED_RANGE for d in dofs}
        elif dofs is None:
            dofs = {}
        self.graph.add_edge(u, v, bone=Bone(tip_to_base, dof_dict=dofs))
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

    from . import draw
    from .utils import make_tip_to_base

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

    kin = body.kinematics().cuda()
    utheta = kin.log_angles_rest_pose()
    poses = kin.fk(utheta).cpu()
    print(poses)

    # poses = [body.fk()]

    # body["root", "a"].set_delta([np.pi / 4, 0, 0, 2.0, 0, 0])
    # body["c", "d"].set_delta([0, 0, -np.pi / 2, 0, 0, 0])
    # body["c", "e"].set_delta([0, 0, -np.pi / 2, 0, 0, 0])
    # poses.append(body.fk())

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
    # draw.draw_kinematics(
    #     ax0,
    #     body=body,
    #     fk=poses[1],
    #     draw_vertex_labels=True,
    #     draw_local_frames=True,
    #     draw_root=True,
    # )
    plt.show()
