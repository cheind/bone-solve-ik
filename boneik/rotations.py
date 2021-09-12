from typing import Tuple, Optional, List
import torch
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
        scale[1, 1] = (length) / 2

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
    return torch.stack(f, 0), torch.stack(i, 0)


def rodrigues(z: torch.FloatTensor, axes: torch.FloatTensor) -> torch.FloatTensor:
    # triu_idx = torch.triu_indices(row=3, col=3, offset=0)
    # aux = x.triu(diagonal=1)
    # aux = aux - aux.t()     # aux is a skew-symmetric matrix
    pass


# exp(z, aff, axes) -> (BxMx2, Mx3x3, M)
