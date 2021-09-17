from typing import Tuple, Optional, List
import torch
import torch.nn.functional as F
import numpy as np

PI = torch.tensor(np.pi).float()
UNCONSTRAINED_RANGE = torch.tensor([-PI, PI]).float()


def affine_constraint(r: torch.FloatTensor) -> torch.FloatTensor:
    """Returns the affine transformation used to constrain an angle to an open interval."""
    r = torch.as_tensor(r).float()
    length = r[1] - r[0]
    assert length > 0, "Upper limit must be greater than lower limit"
    if abs(length - 2 * PI) < 1e-6:
        return torch.eye(3), torch.eye(3)
    else:
        assert length <= PI, "Constrained interval must be <= PI or exactly 2PI"

        theta = length / 2
        scale = torch.eye(3, dtype=r.dtype, device=r.device)
        scale[0, 0] = (1 - torch.cos(theta)) / 2
        scale[1, 1] = torch.sin(theta)

        trans = torch.eye(3, dtype=r.dtype, device=r.device)
        trans[0, 2] = -(-1) * scale[0, 0] + torch.cos(theta)
        trans[1, 2] = -(-1) * scale[1, 1] + -torch.sin(theta)

        rot = torch.eye(3, dtype=r.dtype, device=r.device)
        theta_diff = r[0] - (-theta)
        rot[0, 0] = torch.cos(theta_diff)
        rot[1, 1] = torch.cos(theta_diff)
        rot[0, 1] = -torch.sin(theta_diff)
        rot[1, 0] = torch.sin(theta_diff)

        aff = rot @ trans @ scale
        return aff, torch.inverse(aff)


def affine_constraint_transformations(
    open_ranges: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Returns forward inverse affine transformations associated with given range constraints.

    Params
    ------
    open_ranges: (N,2) tensor
        Open range boundaries for N angles

    Returns
    -------
    affines: (N,3,3) tensor
    inverses: (N,3,3) tensor

    """
    forwards, invs = [], []
    for r in open_ranges:
        f, i = affine_constraint(r)
        forwards.append(f)
        invs.append(i)
    return torch.stack(forwards, 0), torch.stack(invs, 0)


def skew(v: torch.FloatTensor) -> torch.FloatTensor:
    """Convert vectors to cross product matrices..

    Params
    ------
    v: (N,3) tensor
        vectors

    Returns
    -------
    s: (N,3,3) tensor
        Corresponding cross product matrices.
    """

    N, C = v.shape
    assert C == 3

    s = v.new_zeros((N,) + (3, 3))
    x, y, z = v.unbind(-1)

    s[:, 0, 1] = -z
    s[:, 0, 2] = y
    s[:, 1, 0] = z
    s[:, 1, 2] = -x
    s[:, 2, 0] = -y
    s[:, 2, 1] = x

    return s


def rodrigues(z: torch.FloatTensor, axes: torch.FloatTensor) -> torch.FloatTensor:
    """Batch convert angle parametrization to rotation matrices.

    Params
    ------
    z: (B,N,2) tensor
        Unit-length vectors interpreted as complex numbers cos(theta) +isin(theta)
    axes: (N,3) tensor
        Unit-length rotation axes.

    Returns
    -------
    rot: (B,N,3,3) tensor
        Rotation matrices
    """
    B, N, C = z.shape
    assert C == 2
    assert axes.shape == (N, 3)

    cos_theta = z[..., 0]
    sin_theta = z[..., 1]
    C = skew(axes)
    CC = torch.matmul(C, C)
    eye = torch.eye(3, device=z.device, dtype=z.dtype).view(1, 1, 3, 3)

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
    """Convertes unconstrained 2D reals to angle."""
    z = _exp_map(uz, constraints)
    return torch.atan2(z[..., 1], z[..., 0])


def _exp_map(
    uz: torch.FloatTensor, constraints: torch.FloatTensor
) -> torch.FloatTensor:
    """Maps unconstrained 2D reals to constrainted space on unit circle

    Params
    -------
    uz: (B,N,2) tensor
        Batch angle parametrizations
    constraints: (N,3,3) tensor
        Affine angle range constraints

    Returns
    -------
    z: (B,N,2) tensor
        Affine transformed angle reparametrizations, normalized
        interpretable as z[i,j] = cos(theta) + i sin(theta)
    """
    # Unit box
    z = torch.tanh(uz)
    z = F.pad(z, (0, 1), "constant", 1).unsqueeze(-1)  # (B,N,3,1)
    # Apply constraint

    z = torch.matmul(constraints.unsqueeze(0), z).squeeze(-1)
    # Complex on unit-circle
    z = F.normalize(z[..., :2], dim=-1)
    return z


def clamp_angle(
    theta: torch.FloatTensor, open_ranges: torch.FloatTensor
) -> torch.FloatTensor:
    """Clamps angles to nearest in open interval."""
    B, N = theta.shape
    eps = torch.finfo(theta.dtype).eps

    return torch.clamp(
        theta,
        open_ranges[:, 0] + 2 * eps,
        open_ranges[:, 1] - 2 * eps,
    )


def project_(uz: torch.FloatTensor) -> None:
    """Inplace clamping of unconstrained angles to be close to unit box.
    Done to avoid gradients where tanh is close to zero in optimization."""
    uz.data.clamp_(-3.0, 3.0)
    # Actually never allows uangle to be in arange for which tanh(z) = +/- 1
    # avoiding vanishing gradients. Also means, that interval is open-range on
    # both sides. The more relaxed the clamping, the closer values to the
    # interval we can attain.


def log_map_angle(
    theta: torch.FloatTensor,
    inv_constraints: torch.FloatTensor,
) -> torch.FloatTensor:
    """Converts angles to unconstrained 2D real space."""
    z = torch.stack([torch.cos(theta), torch.sin(theta)], -1)
    return log_map(z, inv_constraints)


def log_map(
    z: torch.FloatTensor, inv_constraints: torch.FloatTensor
) -> torch.FloatTensor:
    z = F.pad(z, (0, 1), "constant", 1).unsqueeze(-1)  # (B,N,3,1)
    z = torch.matmul(inv_constraints.unsqueeze(0), z).squeeze(-1)
    return z[..., :2]
    # zu = torch.atanh(z[..., :2])
    return z
