import matplotlib.pyplot as plt
import transformations as T
import torch

from boneik import bones, solvers


def main():
    root = bones.Bone("0")
    b1 = bones.Bone("1", t=torch.Tensor(T.translation_matrix([0, 1.5, 0])))
    b2 = bones.Bone("2", t=torch.Tensor(T.translation_matrix([0, 1.0, 0])))
    b3 = bones.Bone("end", t=torch.Tensor(T.translation_matrix([0, 0.5, 0])))
    root.link_to(b1).link_to(b2).link_to(b3)

    dof_dict = {
        root: bones.BoneDOF(rotz=bones.RotZ()),
        b1: bones.BoneDOF(rotz=bones.RotZ()),
        b2: bones.BoneDOF(rotz=bones.RotZ()),
    }
    solver = solvers.IKSolver(root, dof_dict=dof_dict)
    fig, ax = plt.subplots()

    def on_move(event):
        if not event.inaxes:
            return
        loc = torch.tensor([event.xdata, event.ydata, 0]).float()
        solver.solve(anchor_dict={b3: loc}, lr=1e-0)

        ax = event.inaxes
        with torch.no_grad():
            fk_dict = bones.fk(solver.root, solver.dof_dict)
            for bone in bones.bfs(solver.root):
                if bone in dof_dict:
                    print(
                        bone.name,
                        dof_dict[bone].rotz.angle,
                        dof_dict[bone].rotz.uangle,
                    )
                tb = fk_dict[bone][:2, 3].numpy()
                if bone.parent is not None:
                    tp = fk_dict[bone.parent][:2, 3].numpy()
                    ax.plot([tp[0], tb[0]], [tp[1], tb[1]], c="green")
                ax.scatter([tb[0]], [tb[1]], c="green")
        fig.canvas.draw()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1, 5)
    plt.show()

    pass


if __name__ == "__main__":
    main()