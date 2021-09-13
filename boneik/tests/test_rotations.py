import torch
import numpy as np
from boneik import rotations as R
import transformations as T


def test_rodrigues():
    angles = torch.deg2rad(torch.tensor([0, 90, 180]))
    z = torch.stack((torch.cos(angles), torch.sin(angles)), -1)
    axes = torch.eye(3)
    rot = R.rodrigues(z, axes)

    assert torch.allclose(rot[0], torch.eye(3))
    assert torch.allclose(
        rot[1], torch.tensor([[0, 0, -1], [0, 1.0, 0], [1, 0, 0]]).T, atol=1e-3
    )
    assert torch.allclose(
        rot[2], torch.tensor([[-1, 0, 0], [0, -1.0, 0], [0, 0, 1]]).T, atol=1e-3
    )


def test_rodrigues_multidim():
    angles = torch.deg2rad(torch.tensor([0, 90, 180, 270]))
    z = torch.stack((torch.cos(angles), torch.sin(angles)), -1)
    axes = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    )
    rot = R.rodrigues(z.view(4, 1, 2), axes.view(4, 1, 3))
    assert rot.shape == (4, 1, 3, 3)
    assert torch.allclose(rot[0, 0], torch.eye(3))
    assert torch.allclose(
        rot[1, 0], torch.tensor([[0, 0, -1], [0, 1.0, 0], [1, 0, 0]]).T, atol=1e-3
    )
    assert torch.allclose(
        rot[2, 0], torch.tensor([[-1, 0, 0], [0, -1.0, 0], [0, 0, 1]]).T, atol=1e-3
    )
    assert torch.allclose(
        rot[3, 0], torch.tensor([[0, -1, 0], [1.0, 0, 0], [0, 0, 1]]).T, atol=1e-3
    )


def test_exp_map():
    # unconstrained
    c, cinv = R.range_constraints([None])
    angles = torch.deg2rad(torch.tensor([90]))
    z = torch.stack((torch.cos(angles), torch.sin(angles)), -1)
    axes = torch.tensor([[1.0, 0, 0]])
    rot = R.exp_map(z, axes, c)
    assert torch.allclose(rot[0], torch.tensor([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]]).T)

    # constrained
    c, cinv = R.range_constraints([(-np.pi / 4, np.pi / 4)])

    z = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [-1.0, 1.0]]) * 10
    axes = torch.tensor([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]])
    rot = R.exp_map(z, axes, c)

    rot_trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    phi_cos = (rot_trace - 1.0) * 0.5
    rot_angles = torch.acos(phi_cos)
    assert torch.allclose(
        torch.tensor([np.pi / 4, 0, np.pi / 4]), rot_angles
    )  # note, sign in rot-axis

    # constrained, rotated
    c, cinv = R.range_constraints([(-np.pi / 4 + np.pi, np.pi / 4 + np.pi)])

    z = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [-1.0, 1.0]]) * 10
    axes = torch.tensor([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]])
    rot = R.exp_map(z, axes, c)

    rot_trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    phi_cos = (rot_trace - 1.0) * 0.5
    rot_angles = torch.acos(phi_cos)
    assert torch.allclose(
        torch.tensor([-np.pi / 4 + np.pi, np.pi, np.pi - np.pi / 4]), rot_angles
    )  # note, sign in rot-axis