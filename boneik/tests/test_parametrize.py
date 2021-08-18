import torch
import numpy as np


from boneik.reparametrize import PeriodicAngleReparametrization

PI = np.pi
PI2 = 2 * PI


def constrain(z: torch.Tensor, low: float = -PI, high: float = PI):
    a = torch.angle(torch.view_as_complex(z)) + PI2
    c = a * (high - low) / PI2 + low
    return c


def unconstrain(c: torch.Tensor, low: float = -PI, high: float = PI):
    a = (c - low) * PI2 / (high - low)
    return torch.tensor([torch.cos(a), torch.sin(a)])


def test_periodic_angle_parametrization():
    eps = np.finfo(np.float32).eps
    theta = torch.linspace(-PI + eps, PI - eps, 100)
    z = torch.stack([torch.cos(theta), torch.sin(theta)], -1)
    # z = z.requires_grad_(True)
    reparam = PeriodicAngleReparametrization()
    a = reparam(z)
    assert torch.allclose(theta, a, atol=1e-4)

    theta = torch.linspace(-0.1 + eps, 0.1 - eps, 100)
    reparam = PeriodicAngleReparametrization((-0.1, 0.1))
    z = reparam.inv(theta)
    # The complex numbers should cover 2pi.
    aa = torch.angle(torch.view_as_complex(z))
    assert torch.allclose(aa[-1] - aa[0], torch.tensor([2 * np.pi]))
    a = reparam(reparam.inv(theta))
    assert torch.allclose(theta, a, atol=1e-4)
