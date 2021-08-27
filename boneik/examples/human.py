from boneik.reparametrizations import PI
import matplotlib.pyplot as plt
import transformations as T
import torch
import pickle

from boneik import kinematics, solvers


def make_tuv(length: float, axis_in_parent: str):
    alut = {
        "x": torch.tensor([1.0, 0, 0]),
        "y": torch.tensor([0.0, 1.0, 0]),
        "z": torch.tensor([0.0, 0.0, 1.0]),
        "-x": torch.tensor([-1.0, 0, 0]),
        "-y": torch.tensor([0.0, -1.0, 0]),
        "-z": torch.tensor([0.0, 0.0, -1.0]),
    }
    ax, ay, az = axis_in_parent.split(",")
    r = torch.eye(4)
    r[:3, 0] = alut[ax]
    r[:3, 1] = alut[ay]
    r[:3, 2] = alut[az]
    t = torch.eye(4)
    t[:3, 3] = torch.tensor([0, length, 0])
    return r @ t  # note, reversed to have translation around local frame.


def draw_axis(ax3d, t_world: torch.Tensor, length: float = 0.5, lw: float = 1.0):
    p_f = torch.tensor(
        [[0, 0, 0, 1], [length, 0, 0, 1], [0, length, 0, 1], [0, 0, length, 1]]
    ).T
    p_0 = t_world @ p_f
    print(p_0)
    X = torch.stack([p_0[:, 0].T, p_0[:, 1].T], 0).numpy()
    Y = torch.stack([p_0[:, 0].T, p_0[:, 2].T], 0).numpy()
    Z = torch.stack([p_0[:, 0].T, p_0[:, 3].T], 0).numpy()
    print(X)
    ax3d.plot3D(X[:, 0], X[:, 1], X[:, 2], "r-", linewidth=lw)
    ax3d.plot3D(Y[:, 0], Y[:, 1], Y[:, 2], "g-", linewidth=lw)
    ax3d.plot3D(Z[:, 0], Z[:, 1], Z[:, 2], "b-", linewidth=lw)


@torch.no_grad()
def draw(
    ax3d,
    graph: kinematics.SkeletonGraph,
    anchors: torch.Tensor = None,
    draw_local_frames: bool = True,
    draw_vertex_labels: bool = False,
):
    fkt = kinematics.fk(graph)
    root = graph.graph["root"]
    for u, v in graph.graph["bfs_edges"]:
        x = fkt[[u, v], 0, 3].numpy()
        y = fkt[[u, v], 1, 3].numpy()
        z = fkt[[u, v], 2, 3].numpy()
        ax3d.plot(x, y, z, lw=1, c="k", linestyle="--")
        ax3d.scatter(x[1], y[1], z[1], c="k")
    if draw_local_frames:
        draw_axis(ax3d, fkt[root])
        for _, v in graph.graph["bfs_edges"]:
            draw_axis(ax3d, fkt[v])

    if anchors is not None:
        anchors = anchors.numpy()
        ax3d.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], c="r", marker="o")


def main():
    g = kinematics.SkeletonGenerator()
    g.bone("torso", "chest", make_tuv(1.17965, "-x,z,y"))
    g.bone("chest", "neck", make_tuv(2.0279, "x,y,z"))
    g.bone("neck", "head", make_tuv(0.73577, "x,y,z"))

    g.bone("neck", "shoulder.L", make_tuv(0.71612, "-z,-x,y"))
    g.bone("shoulder.L", "elbow.L", make_tuv(1.8189, "x,y,z"))
    g.bone("elbow.L", "hand.L", make_tuv(1.1908, "x,y,z"))

    g.bone("neck", "shoulder.R", make_tuv(0.71612, "z,x,y"))
    g.bone("shoulder.R", "elbow.R", make_tuv(1.8189, "x,y,z"))
    g.bone("elbow.R", "hand.R", make_tuv(1.1908, "x,y,z"))

    g.bone("torso", "hip.R", make_tuv(1.1542, "-y,x,z"))
    g.bone("hip.R", "knee.R", make_tuv(2.2245, "x,-z,y"))
    g.bone("knee.R", "foot.R", make_tuv(1.7149, "x,y,z"))

    g.bone("torso", "hip.L", make_tuv(1.1542, "y,-x,z"))
    g.bone("hip.L", "knee.L", make_tuv(2.2245, "x,-z,y"))
    g.bone("knee.L", "foot.L", make_tuv(1.7149, "x,y,z"))

    g.bone("root", "torso", torch.eye(4), rotz=kinematics.RotZ())

    graph = g.create_graph(
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
    N = graph.number_of_nodes()
    frame_data = pickle.load(open(r"C:\dev\bone-solve-ik\etc\frames.pkl", "rb"))
    anchors = torch.zeros((N, 3))
    weights = torch.ones(N)
    weights[-1] = 0
    anchors[: N - 1] = torch.from_numpy(frame_data[0]).float()

    solver = solvers.IKSolver(graph)
    solver.solve(anchors, weights)
    print(kinematics.fmt_skeleton(graph))

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    draw(ax, graph, anchors)

    ax.set_xlim(-5.0, 5.0)
    ax.set_xlabel("x")
    ax.set_ylim(-5.0, 5.0)
    ax.set_ylabel("y")
    ax.set_zlim(-5.0, 5.0)
    ax.set_zlabel("z")
    plt.show()

    print(kinematics.fk(graph))


# "head", 'neck', 'shoulder.R', 'elbow.R', 'hand.R', # 0-4
# 'shoulder.L', 'elbow.L', 'hand.L', # 5-7
# 'hip.R', 'knee.R', 'foot.R', 'hip.L', 'knee.L', 'foot.L', # 8-13
# 'hip', 'chest']

if __name__ == "__main__":
    main()