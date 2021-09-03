import torch
import numpy as np
from . import kinematics


def make_tuv(length: float, axis_in_parent: str):
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


def make_dofs(**kwargs):
    a = kwargs.pop("rx", None)
    i = kwargs.pop("irx", None)
    rx = None
    if a is not None:
        if i is not None:
            i = tuple([np.deg2rad(a) for a in i])
        rx = kinematics.RotX(angle=np.deg2rad(a), interval=i)

    a = kwargs.pop("ry", None)
    i = kwargs.pop("iry", None)
    ry = None
    if a is not None:
        if i is not None:
            i = tuple([np.deg2rad(a) for a in i])
        ry = kinematics.RotY(angle=np.deg2rad(a), interval=i)

    a = kwargs.pop("rz", None)
    i = kwargs.pop("irz", None)
    rz = None
    if a is not None:
        if i is not None:
            i = tuple([np.deg2rad(a) for a in i])
        rz = kinematics.RotZ(angle=np.deg2rad(a), interval=i)

    o = kwargs.pop("tx", None)
    tx = None
    if o is not None:
        tx = kinematics.TransX(offset=o)

    o = kwargs.pop("ty", None)
    ty = None
    if o is not None:
        ty = kinematics.TransY(offset=o)

    o = kwargs.pop("tz", None)
    tz = None
    if o is not None:
        tz = kinematics.TransZ(offset=o)

    assert len(kwargs) == 0, f"Unknown kwargs {list(kwargs.keys())}"

    return dict(rx=rx, ry=ry, rz=rz, tx=tx, ty=ty, tz=tz)
