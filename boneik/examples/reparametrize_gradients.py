import torch
import matplotlib.pyplot as plt

from boneik.reparametrize import PeriodicAngleReparametrization, PI


def compute_partial_diffs(a, z):
    """Returns the partial derivatives of angles with respect to real/im part of unconstrained complex number representation."""
    grad = torch.stack(
        [
            torch.autograd.grad(a[i], z, retain_graph=True)[0][i].detach()
            for i in range(len(z))
        ],
        0,
    )
    return grad  # Nx2


def main():
    interval = (-PI, PI)
    theta = torch.linspace(interval[0], interval[1], 100)
    reparam = PeriodicAngleReparametrization(interval)
    z = reparam.inv(theta)
    z.requires_grad_(True)
    a = reparam(z)
    grads = compute_partial_diffs(a, z)

    fig, axs = plt.subplots(2, 1, sharex=True)
    print(z[:, 0])
    axs[0].plot(z[:50, 0].detach(), grads[:50, 0])
    axs[0].plot(z[50:, 0].detach(), grads[50:, 0])
    axs[1].plot(z[:, 1].detach(), grads[:, 1])
    plt.show()


if __name__ == "__main__":
    main()
