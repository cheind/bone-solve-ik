import torch
import numpy as np
from boneik import rotations as R
import transformations as T


def test_rodrigues():
    angles = torch.deg2rad(torch.tensor([0, 90, 180]))
    z = torch.stack((torch.cos(angles), torch.sin(angles)), -1)
    axes = torch.eye(3)
    rot = R.rodrigues(z.view(1, 3, 2), axes)

    assert torch.allclose(rot[0, 0], torch.eye(3))
    assert torch.allclose(
        rot[0, 1], torch.tensor([[0, 0, -1], [0, 1.0, 0], [1, 0, 0]]).T, atol=1e-3
    )
    assert torch.allclose(
        rot[0, 2], torch.tensor([[-1, 0, 0], [0, -1.0, 0], [0, 0, 1]]).T, atol=1e-3
    )


def test_exp_map():
    # unconstrained
    c, _ = R.affine_constraint_transformations(torch.tensor([[-R.PI, R.PI]]))
    angles = torch.deg2rad(torch.tensor([90]))
    z = torch.stack((torch.cos(angles), torch.sin(angles)), -1)
    axes = torch.tensor([[1.0, 0, 0]])
    rot = R.exp_map(z, axes, c)  # Note, c is broadcasted for each of the elements.
    assert torch.allclose(rot[0], torch.tensor([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]]).T)

    # constrained
    c, _ = R.affine_constraint_transformations(torch.tensor([(-np.pi / 4, np.pi / 4)]))

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
    c, _ = R.affine_constraint_transformations(
        torch.tensor([(-np.pi / 4 + np.pi, np.pi / 4 + np.pi)])
    )

    z = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [-1.0, 1.0]]) * 10
    axes = torch.tensor([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]])
    rot = R.exp_map(z, axes, c)

    rot_trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    phi_cos = (rot_trace - 1.0) * 0.5
    rot_angles = torch.acos(phi_cos)
    assert torch.allclose(
        torch.tensor([-np.pi / 4 + np.pi, np.pi, np.pi - np.pi / 4]), rot_angles
    )  # note, sign in rot-axis

    # two different constraints
    c, _ = R.affine_constraint_transformations(
        torch.tensor([(-R.PI, R.PI), (-np.pi / 4, np.pi / 4)])
    )
    z = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]]) * 10
    axes = torch.tensor([[1.0, 0, 0], [1.0, 0, 0]])
    rot = R.exp_map(z, axes, c)
    rot_trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    phi_cos = (rot_trace - 1.0) * 0.5
    rot_angles = torch.acos(phi_cos)
    assert torch.allclose(rot_angles, torch.tensor([3 * np.pi / 4, np.pi / 4]))


def test_exp_map_angle():
    c, _ = R.affine_constraint_transformations(torch.tensor([(-np.pi / 4, np.pi / 4)]))
    z = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [-1.0, 1.0]]) * 10
    assert torch.allclose(
        R.exp_map_angle(z, c), torch.tensor([-np.pi / 4, 0, np.pi / 4])
    )

    c, _ = R.affine_constraint_transformations(
        torch.tensor([(-np.pi / 4 + np.pi, np.pi / 4 + np.pi)])
    )
    z = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [-1.0, 1.0]]) * 10
    assert torch.allclose(
        R.exp_map_angle(z, c),
        torch.tensor([3 / 4 * np.pi, np.pi, -np.pi + np.pi / 4]),
    )

    c, _ = R.affine_constraint_transformations(
        torch.tensor([(-R.PI, R.PI), (-np.pi / 4, np.pi / 4)])
    )
    z = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]]) * 10
    assert torch.allclose(
        R.exp_map_angle(z, c),
        torch.tensor([-3 / 4 * np.pi, -np.pi / 4]),
    )


def test_clamp_angle():
    theta = torch.tensor([-0.6, 0.2, 0.3, 0.5, 0.1]).unsqueeze(0)
    r = torch.tensor([[-0.3, 0.3], [-0.5, 0.1], [0.0, 0.5], [0.0, 0.1], [0.0, 0.1]])
    thetac = R.clamp_angle(theta, r)
    assert torch.allclose(thetac, torch.tensor([-0.3, 0.1, 0.3, 0.1, 0.1]).unsqueeze(0))


def test_project():
    uz = torch.tensor([[0.7, -0.7]]) * 10
    uz.requires_grad_(True)
    R.project_(uz)
    theta = R.exp_map_angle(uz, torch.eye(3).view(1, 3, 3))
    theta.sum().backward()
    assert abs(uz.grad[0, 0]) > 1e-3
    assert abs(uz.grad[0, 1]) > 1e-3


def test_log_map_angle():
    theta = torch.tensor([0.0, R.PI / 4, -R.PI / 4]).unsqueeze(0)
    r = torch.tensor([(-R.PI, R.PI), (-R.PI, R.PI), (-R.PI, R.PI)])
    theta = R.clamp_angle(theta, r)
    c, cinv = R.affine_constraint_transformations(r)
    uz = R.log_map_angle(theta, cinv)
    thetar = R.exp_map_angle(uz, c)
    assert torch.allclose(theta, thetar)

    theta = torch.tensor([0.0, R.PI / 4, -R.PI / 4]).unsqueeze(0)
    r = torch.tensor([(0.0, R.PI), (-R.PI / 2, R.PI / 2), (-R.PI / 2, R.PI / 2)])
    thetac = R.clamp_angle(theta, r)
    c, cinv = R.affine_constraint_transformations(r)
    uz = R.log_map_angle(thetac, cinv)
    thetar = R.exp_map_angle(uz, c)
    assert torch.allclose(theta, thetar, atol=1e-4)
