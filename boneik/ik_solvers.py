from functools import partial

import torch
import torch.optim as optim
import logging

from . import bodies
from . import ik_criteria
from . import rotations as R

_logger = logging.getLogger("boneik")


def _closure_step(
    kin: bodies.BodyKinematics,
    log_angles: torch.FloatTensor,
    mask: torch.FloatTensor,
    opt: optim.LBFGS,
    crit: ik_criteria.IKCriterium,
):
    masked_angles = log_angles * mask + (1 - mask) * log_angles.detach()
    opt.zero_grad()
    loss = crit(kin, masked_angles)
    loss.backward()
    return loss


def solve_ik(
    kin: bodies.BodyKinematics,
    log_angles: torch.FloatTensor,
    crit: ik_criteria.IKCriterium,
    *,
    max_epochs: int = 100,
    min_abs_change: float = 1e-5,
    lr: float = 1e0,
    history_size: int = 100,
    max_iter: int = 20,
    reproject: bool = True,
) -> float:
    last_loss, loss = 1e10, 1e10
    log_angles.requires_grad_(True)

    # Masked is used to vanish gradients for non-unlocked
    # rotations.
    mask = kin.rot_unlock_mask.unsqueeze(0).unsqueeze(-1).float()

    opt = optim.LBFGS(
        [log_angles],
        history_size=history_size,
        max_iter=max_iter,
        lr=lr,
        line_search_fn="strong_wolfe",
    )
    closure = partial(
        _closure_step, opt=opt, crit=crit, kin=kin, log_angles=log_angles, mask=mask
    )
    for e in range(max_epochs):
        opt.step(closure)
        loss = crit(kin, log_angles).item()
        if (last_loss - loss) < min_abs_change:
            break
        last_loss = loss
        if reproject:
            R.project_(log_angles)
    _logger.debug(f"Completed after {e+1} epochs, loss {loss}")
    return loss


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import transformations as T

    from .reparametrizations import PI
    from . import draw

    b = bodies.BodyBuilder()

    b.add_bone(
        "root",
        "j1",
        tip_to_base=T.translation_matrix([0, 1.0, 0]),
        dofs={"rz": (-PI / 4, PI / 4)},
    ).add_bone(
        "j1",
        "j2",
        tip_to_base=T.translation_matrix([0, 1.0, 0]),
        dofs={"rz": (-PI / 4, -PI / 8)},
    )
    body = b.finalize(["root", "j1", "j2"])
    kin = body.kinematics()

    log_angles = kin.log_angles_rest_pose().requires_grad_(True)

    anchors = torch.zeros(3, 3)
    weights = torch.zeros(3)

    anchors[1] = torch.Tensor([1.0, 1.0, 1.0])
    weights[1] = 0.1

    anchors[2] = torch.Tensor([2.0, 0.0, 0])
    weights[2] = 1.0

    loss = solve_ik(
        kin,
        log_angles,
        ik_criteria.EuclideanDistanceCriterium(anchors, weights),
        lr=1e-1,
    )
    print(body)
    print("loss", loss)
    pose = kin.fk(log_angles).squeeze(0)
    print(pose)

    fig, ax = draw.create_figure3d(axes_ranges=[[-2, 2], [-2, 2], [0, 1]])
    draw.draw_kinematics(ax, body=body, fk=pose, anchors=anchors, draw_root=True)
    plt.show()
