from typing import Tuple, Optional
import warnings
import numpy as np
import torch
import torch.nn.functional as F

PI = np.pi
PI2 = 2 * PI


class BoundsViolatedWarning(UserWarning):
    pass


class ConstrainedAngleReparametrization:
    """Reparametrize an contrained angle for unconstrained optimization.

    The unconstrained angle is represented by unbounded 2 reals, (x,y), living
    on the Euclidean plane in R2. The constrained angle, a, is represented
    by the corresponding the unit complex number such that z = cos(a) + i sin(a),
    where x is an angle within a user specified interval.

        z   = f(x,y)
            = norm(affine(tanh(x,y))
    """

    def __init__(self, open_interval: Optional[Tuple[float, float]] = None) -> None:
        if open_interval is None:
            self.i = (-PI, PI)
            self.length = 2 * PI
            self.is_constrained = False
        else:
            self.i = open_interval
            self.length = self.i[1] - self.i[0]
            self.is_constrained = True
        assert (
            open_interval is None or self.length <= PI
        ), "Constrained interval must be <= PI."
        self._init_affine()

    def _init_affine(self):
        if self.is_constrained:
            theta = self.length / 2
            scale = torch.eye(3)
            scale[0, 0] = (1 - np.cos(theta)) / 2
            scale[1, 1] = (self.length) / 2

            trans = torch.eye(3)
            trans[0, 2] = -(-1) * scale[0, 0] + np.cos(theta)
            trans[1, 2] = -(-1) * scale[1, 1] + -np.sin(theta)

            rot = torch.eye(3)
            theta_diff = self.i[0] - (-theta)
            rot[0, 0] = np.cos(theta_diff)
            rot[1, 1] = np.cos(theta_diff)
            rot[0, 1] = -np.sin(theta_diff)
            rot[1, 0] = np.sin(theta_diff)
            self.affine = rot @ trans @ scale
        else:
            self.affine = torch.eye(3)
        self.inv_affine = torch.inverse(self.affine)

    def exp(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(z)
        z = F.pad(z, (0, 1), "constant", 1)
        z = self.affine @ z
        z = F.normalize(z[:2], dim=-1)
        return z

    def log(self, z: torch.Tensor) -> torch.Tensor:
        z = F.pad(z, (0, 1), "constant", 1)
        z = self.inv_affine @ z
        z = torch.atanh(z[:2])
        return z

    def angle2log(self, theta: torch.Tensor) -> torch.Tensor:
        if self.is_constrained and (theta <= self.i[0] or theta >= self.i[1]):
            theta = (self.i[0] + self.i[1]) * 0.5
            warnings.warn(
                f"Angle out of bounds, changing to midpoint {theta:.3f}",
                BoundsViolatedWarning,
            )
        theta = torch.as_tensor(theta)
        z = torch.tensor([torch.cos(theta), torch.sin(theta)])
        return self.log(z)

    def log2angle(self, z: torch.Tensor) -> torch.Tensor:
        e = self.exp(z)
        return torch.atan2(e[1], e[0])

    def project_inplace(self, z: torch.Tensor):
        z.data.clamp_(
            -2.0, 2.0
        )  # Actually never allows uangle to be in arange for which tanh(z) = +/- 1
        # avoiding vanishing gradients. Also means, that interval is open-range on
        # both sides.