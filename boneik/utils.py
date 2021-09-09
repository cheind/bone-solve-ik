import torch


def make_tip_to_base(length: float, axis_in_parent: str):
    alut = {
        "x": torch.tensor([1.0, 0, 0]),
        "y": torch.tensor([0.0, 1.0, 0]),
        "z": torch.tensor([0.0, 0.0, 1.0]),
        "-x": torch.tensor([-1.0, 0, 0]),
        "-y": torch.tensor([0.0, -1.0, 0]),
        "-z": torch.tensor([0.0, 0.0, -1.0]),
    }
    ax, ay, az = axis_in_parent.split(",")
    r = torch.eye(4)
    r[:3, 0] = alut[ax]
    r[:3, 1] = alut[ay]
    r[:3, 2] = alut[az]
    t = torch.eye(4)
    t[:3, 3] = torch.tensor([0, length, 0])
    return r @ t  # note, reversed to have translation around local frame.