from typing import Tuple, Optional, List
import torch
import torch.nn.functional as F
import numpy as np

PI = np.pi
OpenRange = Tuple[float, float]


def _affine(r: Optional[OpenRange]) -> torch.FloatTensor:
    """Returns the affine transformation used to constrain an angle to an open interval."""
    if r is None:
        # -pi/pi
        return torch.eye(3), torch.eye(3)
    else:
        length = r[1] - r[0]
        assert length > 0, "Upper limit must be greater than lower limit"
        assert length <= PI, "Constrained interval must be <= PI"

        theta = length / 2
        scale = torch.eye(3)
        scale[0, 0] = (1 - np.cos(theta)) / 2
        scale[1, 1] = np.sin(theta)

        trans = torch.eye(3)
        trans[0, 2] = -(-1) * scale[0, 0] + np.cos(theta)
        trans[1, 2] = -(-1) * scale[1, 1] + -np.sin(theta)

        rot = torch.eye(3)
        theta_diff = r[0] - (-theta)
        rot[0, 0] = np.cos(theta_diff)
        rot[1, 1] = np.cos(theta_diff)
        rot[0, 1] = -np.sin(theta_diff)
        rot[1, 0] = np.sin(theta_diff)

        aff = rot @ trans @ scale
        return aff, torch.inverse(aff)


def range_constraints(
    open_ranges: List[Optional[OpenRange]],
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Returns forward inverse affine transformations associated with given range constraints.

    Returns
    -------
    affines: Nx3x3 tensor
    inverses: Nx3x3 tensor

    """
    forwards, invs = [], []
    for r in open_ranges:
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
    # Unit box
    z = torch.tanh(uz)
    z = F.pad(z, (0, 1), "constant", 1)
    # Apply constraint
    z = torch.matmul(constraints, z.unsqueeze(-1)).squeeze(-1)
    # Complex on unit-circle
    z = F.normalize(z[..., :2], dim=-1)
    # Exponential map to rotation matrix
    return rodrigues(z, axes)