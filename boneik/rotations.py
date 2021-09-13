from typing import Tuple, Optional, List
import torch
import torch.nn.functional as F
import numpy as np

PI = torch.tensor(np.pi).float()
NO_CONSTRAINT = torch.tensor([-PI, PI]).float()


def _affine(r: torch.FloatTensor) -> torch.FloatTensor:
    """Returns the affine transformation used to constrain an angle to an open interval."""
    length = r[1] - r[0]
    assert length > 0, "Upper limit must be greater than lower limit"
    if abs(length - 2 * PI) < 1e-6:
        return torch.eye(3), torch.eye(3)
    else:
        assert length <= PI, "Constrained interval must be <= PI or exactly 2PI"

        theta = length / 2
        scale = torch.eye(3)
        scale[0, 0] = (1 - torch.cos(theta)) / 2
        scale[1, 1] = torch.sin(theta)

        trans = torch.eye(3)
        trans[0, 2] = -(-1) * scale[0, 0] + torch.cos(theta)
        trans[1, 2] = -(-1) * scale[1, 1] + -torch.sin(theta)

        rot = torch.eye(3)
        theta_diff = r[0] - (-theta)
        rot[0, 0] = torch.cos(theta_diff)
        rot[1, 1] = torch.cos(theta_diff)
        rot[0, 1] = -torch.sin(theta_diff)
        rot[1, 0] = torch.sin(theta_diff)

        aff = rot @ trans @ scale
        return aff, torch.inverse(aff)


def range_constraints(
    open_ranges: List[Optional[Tuple[float, float]]],
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Returns forward inverse affine transformations associated with given range constraints.

    Returns
    -------
    affines: Nx3x3 tensor
    inverses: Nx3x3 tensor

    """
    forwards, invs = [], []
    for r in open_ranges:
        if r is None:
            r = NO_CONSTRAINT
        else:
            r = torch.tensor(r).float()
        f, i = _affine(r)
        forwards.append(f)
        invs.append(i)
    return torch.stack(forwards, 0), torch.stack(invs, 0)


def skew(axes: torch.FloatTensor) -> torch.FloatTensor:
    """Compute the skew-symmetric matrices corresponding to the vector cross product."""

    PRE, D = axes.shape[:-1], axes.shape[-1]
    assert D == 3

    s = axes.new_zeros(PRE + (3, 3))
    x, y, z = axes.unbind(-1)

    s[..., 0, 1] = -z
    s[..., 0, 2] = y
    s[..., 1, 0] = z
    s[..., 1, 2] = -x
    s[..., 2, 0] = -y
    s[..., 2, 1] = x

    return s


def rodrigues(z: torch.FloatTensor, axes: torch.FloatTensor) -> torch.FloatTensor:
    """Computes rotation matrices from angles and axes.

    Params
    ------
    z: (*,2) tensor
        Unit-length vectors interpreted as complex numbers cos(theta) +isin(theta)
    axes: (*,3) tensor
        Unit-length rotation axes.

    Returns
    -------
    rot: (*,3,3) tensor
        Rotation matrices
    """
    PRE = z.shape[:-1]
    assert PRE == axes.shape[:-1]
    assert z.shape[-1] == 2
    assert axes.shape[-1] == 3

    cos_theta = z[..., 0]
    sin_theta = z[..., 1]
    C = skew(axes)
    CC = torch.matmul(C, C)
    eye = torch.eye(3, device=z.device, dtype=z.dtype)[(None,) * len(PRE)]

    return eye + sin_theta[..., None, None] * C + (1 - cos_theta[..., None, None]) * CC


def exp_map(
    uz: torch.FloatTensor, axes: torch.FloatTensor, constraints: torch.FloatTensor
) -> torch.FloatTensor:
    """Constrained exponential map from 2D unconstrained angles to 3D rotations."""
    # unit box -> constrained box
    z = _exp_map(uz, constraints)
    # Exponential map from unit circle to rotation matrix
    return rodrigues(z, axes)


def exp_map_angle(
    uz: torch.FloatTensor, constraints: torch.FloatTensor
) -> torch.FloatTensor:
    z = _exp_map(uz, constraints)
    return torch.atan2(z[..., 1], z[..., 0])


def _exp_map(
    uz: torch.FloatTensor, constraints: torch.FloatTensor
) -> torch.FloatTensor:
    """Maps unconstrained 2D reals to constrainted space on unit circle"""
    # Unit box
    z = torch.tanh(uz)
    z = F.pad(z, (0, 1), "constant", 1)
    # Apply constraint
    z = torch.matmul(constraints, z.unsqueeze(-1)).squeeze(-1)
    # Complex on unit-circle
    z = F.normalize(z[..., :2], dim=-1)
    return z


# def clamp_angle(
#     self, theta: torch.FloatTensor, r: List[Optional[OpenRange]]
# ) -> torch.FloatTensor:
#     """Clamps angles to nearest on open interval."""
#     theta = torch.as_tensor(theta).float()
#     eps = torch.finfo(theta.dtype).eps

#     if self.is_constrained and (theta <= self.i[0] or theta >= self.i[1]):
#         # Limit to open interval

#         theta = torch.clamp(theta, self.i[0] + 2 * eps, self.i[1] - 2 * eps)
#     z = torch.tensor([torch.cos(theta), torch.sin(theta)])
#     return self.log(z)
