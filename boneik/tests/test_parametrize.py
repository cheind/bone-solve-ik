import torch
import numpy as np

PI = np.pi
PI2 = 2 * PI


def constrain(z: torch.Tensor, low: float = -PI, high: float = PI):
    a = torch.angle(torch.view_as_complex(z)) + PI2
    c = a * (high - low) / PI2 + low
    return c


def unconstrain(c: torch.Tensor, low: float = -PI, high: float = PI):
    a = (c - low) * PI2 / (high - low)
    return torch.tensor([torch.cos(a), torch.sin(a)])


def test_angle_parametrization():
    theta = torch.linspace(-PI, PI, 100)
    z = torch.stack([torch.cos(theta), torch.sin(theta)], -1)
    z = z.requires_grad_(True)
    c = constrain(z, low=-0.05, high=0.05)
    # print(torch.autograd.grad(c[0], z))
    plot_gradients(c, z)

    # z = unconstrain(torch.tensor([0]))
    # print(z)
    # print(constrain(z))


def plot_gradients(c, z):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, sharex=True)
    grads = torch.stack(
        [
            torch.autograd.grad(c[i], z, retain_graph=True)[0][i].detach()
            for i in range(len(z))
        ],
        0,
    )
    print(z.shape, grads.shape)
    print(z[:, 0])

    #
    # ]
    axs[0].plot(z[:, 0].detach(), grads[:, 0])
    axs[1].plot(z[:, 1].detach(), grads[:, 1])
    plt.show()
